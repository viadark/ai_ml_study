import kfp
import kfp.components as comp
from kfp import dsl

@dsl.pipeline(
    name='test-boston',
    description='test boston contest'
)

def boston_pipeline():
    add_p = dsl.ContainerOp(
        name="load iris data pipeline",
        image="minseok2.kang/boston-preprocessing:0.1",
        arguments=[
            '--data_path', './boston_contest.csv'
        ],
        file_outputs={'boston' : '/boston.csv'}
    )
    ml = dsl.ContainerOp(
        name="training pipeline",
        image="minseok2.kang/boston-train:0.1",
        arguments=[
            '--data', add_p.outputs['boston']
        ]
    )
    ml.after(add_p)

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(boston_pipeline, __file__ + ".tar.gz")
    