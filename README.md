# java-adaptive-resonance-theory
Package provides java implementation of algorithms in the field of adaptive resonance theory (ART) 

# Install

Add the following dependency to your POM file:

```xml
<dependency>
  <groupId>com.github.chen0040</groupId>
  <artifactId>java-adaptive-resonance-theory</artifactId>
  <version>1.0.6</version>
</dependency>
```

# Features

Algorithms included:

* ART1
* FuzzyART
* ARTMAP

Applications included:

* Clustering (FuzzyART, ART1)
* Multi-class Classification (ARTMAP)
* Reinforcement Learning (FuzzyART)

# Usage

### Multi-class Classification using ARTMAP

To create and train a ARTMAP classifier:

```java
ARTMAPClassifier classifier = new ARTMAPClassifier();
clasifier.fit(trainingData);
```

The "trainingData" is a data frame which holds data rows with labeled output (Please refers to this [link](https://github.com/chen0040/java-data-frame) to find out how to store data into a data frame)

To predict using the trained ARTMAP classifier:

```java
String predicted_label = classifier.transform(dataRow);
```

The detail on how to use this can be found in the unit testing codes. Below is a complete sample codes of classifying on the libsvm-formatted heart-scale data:

```java
InputStream inputStream = new FileInputStream("heart_scale");
DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();

// as the dataFrame obtained thus far has numeric output instead of labeled categorical output, the code below performs the categorical output conversion
dataFrame.unlock();
for(int i=0; i < dataFrame.rowCount(); ++i){
 DataRow row = dataFrame.row(i);
 row.setCategoricalTargetCell("category-label", "" + row.target());
}
dataFrame.lock();

double alpha = 9.89;
double beta = 0.3;
double rho = 0.01;
classifier.setAlpha(alpha);
classifier.setBeta(beta);
classifier.setRho0(rho);

classifier.fit(dataFrame);

for(int i = 0; i < dataFrame.rowCount(); ++i){
  DataRow tuple = dataFrame.row(i);
  String predicted_label = classifier.transform(tuple);
  System.out.println("predicted: "+predicted_label+"\tactual: "+tuple.categoricalTarget());
}

```

### Spatial Segmentation (Clustering) using ART1

The following sample code shows how to do clustering using ART1:

```java
DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
      .newInput("c1")
      .newInput("c2")
      .newOutput("designed")
      .end();

Sampler.DataSampleBuilder negativeSampler = new Sampler()
      .forColumn("c1").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? 2 : 4))
      .forColumn("c2").generate((name, index) -> randn() * 0.3 + (index % 2 == 0 ? 2 : 4))
      .forColumn("designed").generate((name, index) -> 0.0)
      .end();

Sampler.DataSampleBuilder positiveSampler = new Sampler()
      .forColumn("c1").generate((name, index) -> rand(-4, -2))
      .forColumn("c2").generate((name, index) -> rand(-2, -4))
      .forColumn("designed").generate((name, index) -> 1.0)
      .end();

DataFrame data = schema.build();

data = negativeSampler.sample(data, 200);
data = positiveSampler.sample(data, 200);

System.out.println(data.head(10));

ART1Clustering algorithm = new ART1Clustering();

DataFrame learnedData = algorithm.fitAndTransform(data);

for(int i = 0; i < learnedData.rowCount(); ++i){
 DataRow tuple = learnedData.row(i);
 String clusterId = tuple.getCategoricalTargetCell("cluster");
 System.out.println("learned: " + clusterId +"\tknown: "+tuple.target());
}
```

### Image Segmentation (Clustering) using FuzzyART

The following sample code shows how to use FuzzyART to perform image segmentation:

```java
BufferedImage img= ImageIO.read(FileUtils.getResource("1.jpg"));

DataFrame dataFrame = ImageDataFrameFactory.dataFrame(img);

FuzzyARTClustering cluster = new FuzzyARTClustering();

DataFrame learnedData = cluster.fitAndTransform(dataFrame);

for(int i=0; i <learnedData.rowCount(); ++i) {
 ImageDataRow row = (ImageDataRow)learnedData.row(i);
 int x = row.getPixelX();
 int y = row.getPixelY();
 String clusterId = row.getCategoricalTargetCell("cluster");
 System.out.println("cluster id for pixel (" + x + "," + y + ") is " + clusterId);
}
```

The segmented image can be generated using the trained KMeans from above as illustrated by the following sample code:

```java

List<Integer> classColors = new ArrayList<Integer>();
for(int i=0; i < 5; ++i){
 for(int j=0; j < 5; ++j){
    classColors.add(ImageDataFrameFactory.get_rgb(255, rand.nextInt(255), rand.nextInt(255), rand.nextInt(255)));
 }
}

BufferedImage segmented_image = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
for(int x=0; x < img.getWidth(); x++)
{
 for(int y=0; y < img.getHeight(); y++)
 {
    int rgb = img.getRGB(x, y);

    DataRow tuple = ImageDataFrameFactory.getPixelTuple(x, y, rgb);

    int clusterIndex = cluster.transform(tuple);

    rgb = classColors.get(clusterIndex % classColors.size());

    segmented_image.setRGB(x, y, rgb);
 }
}
```

### Reinforcement Learning 

There is also an example of reinforcement learning (TD-learning using Q-Learning and SARSA) based on FuzzyART. It
is known as "FALCON A Fusion Architecture for Learning Cognition and Navigation", and the sample codes can be found
in the [src/test/java/com/github/chen0040/art/falcon/simulation/minefield](src/test/java/com/github/chen0040/art/rl/simulation/minefield)
The reinforcement learning objective is to navigate a tank (the agent) to a target flag in a mine field, sensors are 
available to the tank when making decision to turn and move forward, immediate reward and delayed rewards were given
to the tank during the Q-Learning and SARSA reinforcement learning process. To launch the reinforcement learning, 
right-click [MineFieldSimulatorGUI.java](src/test/java/com/github/chen0040/art/rl/simulation/minefield/gui/MineFieldSimulatorGUI.java)
and select "Run in main()" in the IntelliJ editor popup menu (or something similar in eclipse or other editors), in the
GUI launched, select "File->Start Simulation" (slow training mode) or "File->Start Simulation (No GUI)" (fast training
mode)