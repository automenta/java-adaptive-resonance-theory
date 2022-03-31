package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * alternative Art impl from
 * https://github.com/thanhld94/Reinforcement-Learning
 *
 * TODO is this ART1 or ART2?
 */
class Art {

    private static final double RO = 0.9;
    private static final double ALPHA = 0.1;

    private int noCategories;
    private final int vectorSize;
    private final ArrayList<double[]> weight;
    private final ArrayList<Boolean> uncommitedNode;
    private final ArrayList<DoubleIntPair> choiceVector;


    public Art(int vSize) {
        vectorSize = vSize;
        noCategories = 0;
        weight = new ArrayList<>();
        uncommitedNode = new ArrayList<>();
        choiceVector = new ArrayList<>();
        addUncommitedNode();
    }

    public int learn(double[] normalizedInput) {
        calChoiceVector(normalizedInput);
        while (true) {
			/*System.out.print( "Input = " );
			for ( int i = 0; i < normalizedInput.length; i++ )
				System.out.printf( "%5.2f ", normalizedInput[ i ] );
			System.out.println();*/

            Collections.sort(choiceVector);
            int category = choiceVector.get(0).index;
            if (vigilanceTest(normalizedInput, weight.get(category)) >= RO) {
                if (uncommitedNode.get(category)) {
                    uncommitedNode.set(category, false);
                    addUncommitedNode();
                }
                weight.set(category, fuzzyAnd(weight.get(category), normalizedInput));
                //System.out.println( "-> Category = " + category );
                return category;
            } else {
                choiceVector.get(0).reset();
            }
        }
    }


    private static double vigilanceTest(double[] i, double[] w) {
        return l1Norm(fuzzyAnd(i, w)) / l1Norm(i);
    }

    private void calChoiceVector(double[] input) {
        for (int j = 0; j < noCategories; j++) {
            choiceVector.get(j).setCategory(j);
            if (uncommitedNode.get(j))
                choiceVector.get(j).setVal((1.0) * vectorSize / (ALPHA + 2 * vectorSize));
            else
                choiceVector.get(j).setVal(l1Norm(fuzzyAnd(input, weight.get(j))) / (ALPHA + l1Norm(weight.get(j))));
        }
    }

    private static double l1Norm(double[] vector) {
        double result = 0.0;
        for (int i = 0; i < vector.length; i++)
            result += vector[i];
        return result;
    }

    private static double[] fuzzyAnd(double[] v1, double[] v2) {
        //System.out.println( "Fuzzy, length = " + v1.length + " " + v2.length );
        double[] result = new double[v1.length];
        for (int i = 0; i < result.length; i++)
            result[i] = min(v1[i], v2[i]);
        return result;
    }

    private static double min(double a, double b) {
        return a <= b ? a : b;
    }

    private void addUncommitedNode() {
        noCategories++;
        uncommitedNode.add(true);
        choiceVector.add(new DoubleIntPair());

        double[] w = new double[vectorSize];
        Arrays.fill(w, 1);
        weight.add(w);
    }


    private static class DoubleIntPair implements Comparable<DoubleIntPair> {

        private double value;
        private int index;

        DoubleIntPair() {
            this(-1, -1);
        }

        DoubleIntPair(double val, int idx) {
            value = val;
            index = idx;
        }

        void setVal(double val) {
            value = val;
        }

        void setCategory(int j) {
            index = j;
        }

        void reset() {
            value = -1;
        }

        @Override
        public int compareTo(DoubleIntPair x) {
            return Double.compare(x.value, this.value);
        }

    }
}