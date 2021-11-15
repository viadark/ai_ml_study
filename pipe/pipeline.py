import kfp
import kfp.components as comp
from kfp import dsl

@dsl.pipeline(
    name='test-boston',
    description='test boston contest'
)

def boston_pipeline():
    add_p = dsl.ContainerOp(
        name="load boston data pipeline",
        image="normalboot/boston-preprocessing:latest",
        arguments=[
            '--data_path', './boston_contest_fillna.csv'
        ],
        file_outputs={'boston' : '/boston_contest.csv'}
    )

    ml = dsl.ContainerOp(
        name="training pipeline",
        image="normalboot/boston-train:latest",
        arguments=[
            '--data', add_p.outputs['boston']
        ],
        file_outputs={'trained_coef': '/trained_coef', 'trained_intercept': '/trained_intercept'}
    )

    ans_set = dsl.ContainerOp(
        name="load answer set data pipeline",
        image="normalboot/boston-preprocessing:latest",
        arguments=[
            '--data_path', './answer.csv'
        ],
        file_outputs={'answer': '/answer.csv'}
    )
    
    test = dsl.ContainerOp(
        name="test pipeline",
        image="normalboot/boston-test:latest",
        arguments=[
            '--answer_data', ans_set.outputs['answer'],
            '--trained_coef', ml.outputs['trained_coef'],
            '--trained_intercept', ml.outputs['trained_intercept']
        ],
        file_outputs={'model': '/model.pkl'}
    )

    ml.after(add_p)
    test.after(ans_set)
    test.after(ml)

if __name__ == "__main__":
    USING_GITHUB_ACTION = False
    if USING_GITHUB_ACTION:
        host = "https://123e4a75d6674638-dot-asia-east1.pipelines.googleusercontent.com"
        pipeline_name = "boston-contest-pipeline"
        namespace = "boston"
        pipeline_package_path = "pipeline.zip"
        version = "v0.1"

        experiment_name = "for dev"
        run_name = "for dev run"

        client = kfp.Client(host=host, namespace=namespace)
        import kfp.compiler as compiler
        compiler.Compiler().compile(boston_pipeline, pipeline_package_path)
        pipeline_id = client.get_pipeline_id(pipeline_name)
        if pipeline_id:
            client.upload_pipeline_version(pipeline_package_path=pipeline_package_path, pipeline_version_name=version, pipeline_name=pipeline_name)
        else:
            client.upload_pipeline(pipeline_package_path=pipeline_package_path, pipeline_name=pipeline_name)
        experiment = client.create_experiment(name=experiment_name, namespace=namespace)
        run = client.run_pipeline(experiment.id, run_name, pipeline_package_path)
    else:
        compiler.Compiler().compile(boston_pipeline, __file__ + ".tar.gz")