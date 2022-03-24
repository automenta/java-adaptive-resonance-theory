package com.github.chen0040.art.classify;


import com.github.chen0040.art.utils.FileUtils;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import org.junit.Test;

import static org.junit.Assert.assertTrue;


/**
 * Created by xschen on 23/5/2017.
 */
public class ARTMAPClassifierUnitTest {

    @Test
    public void TestHeartScale() {

        DataFrame dataFrame = DataQuery.libsvm().from(FileUtils.getResource("heart_scale")).build();

        dataFrame.unlock();

        int rows = dataFrame.rowCount();
        for (int i = 0; i < rows; ++i) {
            DataRow row = dataFrame.row(i);
            row.setCategoricalTargetCell("category-label", String.valueOf(row.target()));
            //System.out.println(row);
        }
        dataFrame.lock();


        double best_alpha = 0, best_beta = 0, best_rho_base = 0;
        double predictionAccuracy = Double.NEGATIVE_INFINITY;

        for (double alpha = 8; alpha < 10; alpha += 0.1) {
            for (double beta = 0; beta < 0.5; beta += 0.1) {
                for (double rho = 0.01; rho < 0.05; rho += 0.1) {
                    ARTMAPClassifier<String> m = new ARTMAPClassifier<>();

                    m.alpha = alpha;
                    m.beta = beta;
                    m.rho0 = rho;

                    m.put(dataFrame);

                    int correctnessCount = 0;
                    for (int i = 0; i < rows; i++) {
                        DataRow r = dataFrame.row(i);
                        String predicted = m.apply(r);
                        String expected = r.categoricalTarget();
                        correctnessCount += predicted.equals(expected) ? 1 : 0;
                    }

                    double accuracy = (correctnessCount * 100.0 / rows);
                    if (accuracy > predictionAccuracy) {
                        best_alpha = alpha;
                        best_beta = beta;
                        best_rho_base = rho;
                        predictionAccuracy = accuracy;
                    }

                }
            }
        }

        System.out.println("best:\talpha: " + best_alpha + "\tbeta: " + best_beta + "\trho_base: " + best_rho_base);
        System.out.println("accuracy: " + predictionAccuracy);

        assertTrue(predictionAccuracy > 0.75f);
    }
}