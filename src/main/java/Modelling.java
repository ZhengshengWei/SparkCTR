import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhengsheng on 2017/2/24.
 */
public class Modelling {

    public static void main(String[] args) {
        if(args.length < 3 ){
            System.err.println("Num of parameter Invalid!");
            return ;
        }
        String path = args[0];
        String testPath = args[1];
        String modelPath = args[2];
        SparkConf sparkConf = new SparkConf()
                .setAppName("CTR_TEST")
                .setMaster("local[2]");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> data = sc.textFile(path);//训练数据

        //去除第一列
        //click,device_type,C1,C15,C16,banner_pos,banner_pos,site_category
      JavaRDD<String> parsedData = data.map(line -> {
            String[] parts = line.split(",");
            String str = "";
            for(int i = 1; i < parts.length ; i ++){
                str += parts[i] + ",";
            }
            return  str.substring(0, str.length() - 1 );
        });

        //将RDD转为ROW RDD
        JavaRDD<Row> rowRDD = parsedData.map(new Function<String,Row>(){
            @Override
            public Row call(String line) throws Exception {
                String[] splited = line.split(",");
                return RowFactory.create(Integer.valueOf(splited[0])
                        , Integer.valueOf(splited[1])
                        , Integer.valueOf(splited[2])
                        , Integer.valueOf(splited[3])
                        , Integer.valueOf(splited[4])
                        , Integer.valueOf(splited[5])
                        , Integer.valueOf(splited[6])
                        , splited[7]);
            }
        });
        //组装构建dataframe
        double[] weights = {0.7, 0.3};
        Long seed = 42L;
        JavaRDD<Row>[] sampleRows = rowRDD.randomSplit(weights, seed);

        List<StructField> fields = new ArrayList<StructField>();
        fields.add(DataTypes.createStructField("click", DataTypes.IntegerType, false));
        fields.add(DataTypes.createStructField("device_type", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("C1", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("C15", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("C16", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("banner_pos", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("banner_pos1", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("site_category", DataTypes.StringType, true));

        SQLContext sqlContext = new SQLContext(sc);
        StructType schema = DataTypes.createStructType(fields);
        Dataset<Row> trainDS = sqlContext.createDataFrame(sampleRows[0], schema);//训练数据
        Dataset<Row> testDS = sqlContext.createDataFrame(sampleRows[1], schema);//测试数据

        //特征转换（int & type->numerical）数据规范化（标准化）
        // int -> double
        // type (one hot encoding) -> numerical
//        trainDS = trainDS.withColumn("click_double",trainDS.col("click").cast("double"));
//        trainDS = trainDS.drop("click").withColumnRenamed("click_double","click");

        // StringIndexer  oneHotEncoder 将 categorical变量转换为 numerical 变量
        // 如某列特征为星期几、天气等等特征，则转换为七个0-1特征
        // 训练数据
        trainDS = oneHotEncoding(trainDS, "site_category", "category_Vector");
        trainDS = oneHotEncoding(trainDS, "device_type", "device_Vector");
        trainDS = oneHotEncoding(trainDS, "C1", "c1_Vector");
        trainDS = oneHotEncoding(trainDS, "C15", "c15_Vector");
        trainDS = oneHotEncoding(trainDS, "C16", "c16_Vector");
        trainDS = oneHotEncoding(trainDS, "banner_pos", "banner_Vector");


        //测试数据 特征标准化、规范化
        testDS = oneHotEncoding(testDS, "site_category", "category_Vector");
        testDS = oneHotEncoding(testDS, "device_type", "device_Vector");
        testDS = oneHotEncoding(testDS, "C1", "c1_Vector");
        testDS = oneHotEncoding(testDS, "C15", "c15_Vector");
        testDS = oneHotEncoding(testDS, "C16", "c16_Vector");
        testDS = oneHotEncoding(testDS, "banner_pos", "banner_Vector");

        trainDS.show(10);
        testDS.show(10);

        /*转换为特征向量*/
        String[] vectorAsCols = {"device_Vector","c1_Vector","c15_Vector","c16_Vector","banner_Vector","category_Vector"};
        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature");
        trainDS = vectorAssembler.transform(trainDS);
        JavaRDD<Row>  fRdd = trainDS.select("click", "vectorFeature").toJavaRDD();

        //测试数据
        testDS = vectorAssembler.transform(testDS);
        JavaRDD<Row>  tRdd = testDS.select("click", "vectorFeature").toJavaRDD();

        JavaRDD<LabeledPoint> finalData = fRdd.map(line -> {
            int click = line.getInt(0);
            Object obj = line.get(1);
            double[] vecs = null;
            if(obj instanceof SparseVector){
                vecs = ((SparseVector) obj).toArray();
            }else if(obj instanceof  DenseVector){
                vecs = ((DenseVector) obj).toArray();
            }
//            System.out.println(new LabeledPoint(click, Vectors.dense(vecs)));
            return new LabeledPoint(click, Vectors.dense(vecs));
        });

        //test
        JavaRDD<LabeledPoint> testData = tRdd.map(line -> {
            int click = line.getInt(0);
            Object obj = line.get(1);
            double[] vecs = null;
            if(obj instanceof SparseVector){
                vecs = ((SparseVector) obj).toArray();
            }else if(obj instanceof  DenseVector){
                vecs = ((DenseVector) obj).toArray();
            }
            return new LabeledPoint(click, Vectors.dense(vecs));
        });

        //模型代入
        int numIterations = 10; //迭代次数
        LinearRegressionModel model = LinearRegressionWithSGD.train(finalData.rdd(), numIterations);
        RidgeRegressionModel model1 = RidgeRegressionWithSGD.train(finalData.rdd(), numIterations);

        print(testData, model);
        print(testData, model1);


        //test1
//        JavaRDD<Vector> predictData = tRdd.map(line -> {
//            Object obj = line.get(1);
//            double[] vecs = null;
//            if(obj instanceof SparseVector){
//                vecs = ((SparseVector) obj).toArray();
//            }else if(obj instanceof  DenseVector){
//                vecs = ((DenseVector) obj).toArray();
//            }
//            return Vectors.dense(vecs);
//        });
//        System.out.println(model.predict(predictData).take(10));
    }

    public static  Dataset<Row> oneHotEncoding(Dataset<Row> ds, String inputColName, String ouputColName){
        StringIndexer scIndex = new StringIndexer().setInputCol(inputColName).setOutputCol(inputColName + "Temp");
        ds = scIndex.fit(ds).transform(ds);
        OneHotEncoder scEncoder = new OneHotEncoder().setInputCol(scIndex.getOutputCol()).setOutputCol(ouputColName);
        ds = scEncoder.transform(ds);
        ds = ds.drop(inputColName, scIndex.getOutputCol());
        return ds;
    }

    public static void print(JavaRDD<LabeledPoint> parsedData, GeneralizedLinearModel model) {
        JavaPairRDD<Object, Object> valuesAndPreds = parsedData.mapToPair(point -> {
            double prediction = model.predict(point.features()); //用模型预测训练数据
            return new Tuple2<>(prediction, point.label());
        });

        //result: threshold accuracy  分类的阈值（类别的分割）曲线
        System.out.println(new BinaryClassificationMetrics(valuesAndPreds.rdd()).precisionByThreshold().toJavaRDD().collect());

        //recall
        // auc
        // logloss

//        Double MSE = valuesAndPreds.mapToDouble((Tuple2<Double, Double> t) -> Math.pow(t._1() - t._2(), 2)).mean(); //计算预测值与实际值差值的平方值的均值
//        System.out.println(model.getClass().getName() + " training Mean Squared Error = " + MSE);
    }
}
