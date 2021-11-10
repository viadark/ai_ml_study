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
        image="normalboot/boston-preprocessing:0.3",
        arguments=[
            '--data_path', './boston_contest.csv'
        ],
        file_outputs={'boston' : '/boston_contest.csv'}
    )

    ml = dsl.ContainerOp(
        name="training pipeline",
        image="normalboot/boston-train:0.5",
        arguments=[
            '--data', add_p.outputs['boston']
        ],
        file_outputs={'trained_coef': '/traind_coef', 'trained_intercept': '/trained_intercept'}
    )

    ans_set = dsl.ContainerOp(
        name="load answer set data pipeline",
        image="normalboot/boston-preprocessing:0.3",
        arguments=[
            '--data_path', './answer.csv'
        ],
        file_outputs={'answer': '/answer.csv'}
    )
    
    test = dsl.ContainerOp(
        name="test pipeline",
        image="normalboot/boston-test:0.2",
        arguments=[
            '--answer_data', ans_set.outputs['answer'],
            '--trained_coef', ml.outputs['trained_coef'],
            '--trained_intercept', ml.outputs['trained_intercept']
        ]
    )
    ml.after(add_p)
    test.after(ans_set)
    test.after(ml)

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(boston_pipeline, __file__ + ".tar.gz")
    