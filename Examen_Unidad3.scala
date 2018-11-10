//1. Importar una sesion spark(SparkSession)
import org.apache.spark.sql.SparkSession

//2. Utilice las lineas de codigo para reportar errores reducidos.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//3. Cree una instancia de la session spark
val spark = SparkSession.builder().getOrCreate()

//4. importar la libreria de Kmeans para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans

//5. Cargar el dataset de Wholesale Customers Data
val dataset=spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

//6. Seleccionar la siguientes columnas para el conjunto de entrenamiento.
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

//7. Import Vector assembler and vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

//8. Crear un nuevo objeto VectorAssembler para las columnas para conjunto de entrenamiento
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

//9. Utilice el objeto assembler para Transformar feature_data
val training_data = assembler.transform(feature_data).select("features")

//10. Crear un modelo Kmeans con K=3
val kmeans= new KMeans().setK(3)

//11. Evaluar los grupos utilizando WSSSE (Within set sum of square errors)
val model = kmeans.fit(training_data)
val KSSSE = model.computeCost(training_data)

//12.Evaluar LOS RESULTADOS
println(s"Resultado: ${KSSSE} ")
