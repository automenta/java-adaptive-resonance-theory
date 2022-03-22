package com.github.chen0040.art.classify;


import com.github.chen0040.art.utils.FileUtils;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import org.junit.Test;

import java.io.InputStream;

import static org.junit.Assert.assertTrue;


/**
 * Created by xschen on 23/5/2017.
 */
public class ARTMAPClassifierUnitTest {

   @Test
   public void TestHeartScale() {
      InputStream inputStream = FileUtils.getResource("heart_scale");

      DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();

      dataFrame.unlock();
      for(int i=0; i < dataFrame.rowCount(); ++i){
         DataRow row = dataFrame.row(i);
         row.setCategoricalTargetCell("category-label", String.valueOf(row.target()));
         System.out.println(row);
      }
      dataFrame.lock();


      double best_alpha=0, best_beta=0, best_rho_base=0;
      double predictionAccuracy = 0;
      int nodeCount = dataFrame.rowCount();

      for(double alpha = 8; alpha < 10; alpha += 0.1) {
         for(double beta = 0; beta < 0.5; beta += 0.1) {
            for(double rho = 0.01; rho < 0.05; rho += 0.1){
               ARTMAPClassifier classifier = new ARTMAPClassifier();

               classifier.alpha = alpha;
               classifier.beta = beta;
               classifier.rho0 = rho;

               classifier.fit(dataFrame);

               int correctnessCount = 0;
               for(int i = 0; i < nodeCount; i++){
                  DataRow r = dataFrame.row(i);
                  String predicted_label = classifier.transform(r);
                  //System.out.println("predicted: "+predicted_label+"\texpected: "+r.getLabelOutput());
                  correctnessCount += predicted_label.equals(r.categoricalTarget()) ? 1 : 0;
               }

               double accuracy = (correctnessCount * 100.0 / dataFrame.rowCount());
               if((accuracy > predictionAccuracy && nodeCount / (double)classifier.nodeCount() > 0.8) || (accuracy / predictionAccuracy > 0.85 && nodeCount > classifier.nodeCount())){
                  best_alpha = alpha;
                  best_beta = beta;
                  best_rho_base = rho;
                  predictionAccuracy = accuracy;
                  nodeCount = classifier.nodeCount();
               }

               //System.out.println("Prediction accuracy: " + (correctnessCount * 100 / batch.size()));
            }
         }
      }

      System.out.println("QAlpha: "+best_alpha + "\tbeta: "+best_beta+"\trho_base: "+best_rho_base);
      System.out.println("accuracy: "+predictionAccuracy);
      System.out.println("nodeCount: "+nodeCount);
      System.out.println("dbSize: "+dataFrame.rowCount());

      assertTrue(predictionAccuracy > 0.75f);



   }
}