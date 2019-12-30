

pip install pyspark

1. Run in intellij IDEA:
    Edit run configuration -- Environment variables -- add the following variables:
    
    export SPARK_HOME=/Users/zhuohuawu/anaconda/envs/tf2/lib/python3.7/site-packages/pyspark;
    export PYSPARK_DRIVER_PYTHON=/Users/zhuohuawu/anaconda/envs/tf2/bin/python;
    export PYSPARK_PYTHON=/Users/zhuohuawu/anaconda/envs/tf2/bin/python;
    export PYTHONPATH=/Users/zhuohuawu/anaconda/envs/tf2/lib/python3.7/site-packages/;

then run the application. 

2. Run in jupyter notebook

    export SPARK_HOME=/Users/zhuohuawu/anaconda/envs/tf2/lib/python3.7/site-packages/pyspark;
    export PYSPARK_SUBMIT_ARGS="pyspark-shell"
    export PYSPARK_DRIVER_PYTHON=ipython
    export PYSPARK_DRIVER_PYTHON_OPTS='notebook' pyspark
    export PATH="$SPARK_HOME/bin:$PATH"
    jupyter notebook


