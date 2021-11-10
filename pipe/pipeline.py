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
        image="normalboot/boston-preprocessing:0.1",
        arguments=[
            '--data_path', './boston_contest.csv'
        ],
        file_outputs={'boston' : '/boston_contest.csv'}
    )
    ml = dsl.ContainerOp(
        name="training pipeline",
        image="normalboot/boston-train:0.2",
        arguments=[
            '--data', add_p.outputs['boston']
        ]
    )
    
    # test = dsl.ContainerOp(
    #     name="test pipeline",
    #     image="normalboot/boston-test:0.1",
    #     arguments=[
    #         '--test_data', './boston_contest_test.csv',
    #         '--data', ml.outputs['model']
    #     ]
    # )
    ml.after(add_p)
    # test.after(ml)

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(boston_pipeline, __file__ + ".tar.gz")
    