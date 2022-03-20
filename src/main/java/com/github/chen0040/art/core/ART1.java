package com.github.chen0040.art.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ART1 {

    protected double alpha; // choice parameter
    protected double rho; // resonance threshold
    protected double beta; // learning rate
    protected List<double[]> weights;
    protected List<Double> activation_values;
    protected int inputCount;


    public ART1() {
        weights = new ArrayList<>();
        activation_values = new ArrayList<>();
    }

    public ART1(int inputCount, int initialNeuronCount) {
        this.inputCount = inputCount;
        weights = new ArrayList<>();
        activation_values = new ArrayList<>();

        for (int i = 0; i < initialNeuronCount; ++i) {
            addNode();
        }
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public void setRho(double rho) {
        this.rho = rho;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public void addNode() {
        double[] neuron = new double[inputCount];
        Arrays.fill(neuron, 1);
        weights.add(neuron);
        activation_values.add(0.0);
    }

    public void addNode(double[] x) {
//		double[] neuron = new double[inputCount];
//		for(int i=0; i < inputCount; ++i){
//			neuron[i] = x[i];
//		}
        weights.add(x);
        activation_values.add(0.0);
    }

    protected double choice_function(double[] x, int j) {
        double[] W_j = weights.get(j);
        double sum = 0, sum2 = 0;
        for (int i = 0; i < x.length; ++i) {
            double Wji = W_j[i];
            sum += Math.abs(x[i] * Wji); // norm1
            sum2 += Math.abs(Wji); //  norm1
        }
        return sum / (alpha + sum2);
    }

    protected int template_with_max_activation_value() {
        int C = getNodeCount();
        double max_activation_value = Double.NEGATIVE_INFINITY;
        int template_selected = -1;
        for (int i = 0; i < C; ++i) {
            double activation_value = activation_values.get(i);
            if (activation_value > max_activation_value) {
                max_activation_value = activation_value;
                template_selected = i;
            }
        }
        return template_selected;
    }

    public int getNodeCount() {
        return weights.size();
    }

    protected double match_function(double[] x, int j) {
        double[] W_j = weights.get(j);
        double sum = 0, sum2 = 0;
        for (int i = 0; i < x.length; ++i) {
            sum += Math.abs(x[i] * W_j[i]); // norm1
            sum2 += Math.abs(x[i]); //  norm1
        }
        return sum / sum2;
    }

    protected void update_node(double[] x, int j) {
        double[] W_j = weights.get(j);

        for (int i = 0; i < x.length; ++i) {
            double Wji = W_j[i];
            W_j[i] = (1 - beta) * Wji + beta * Wji * x[i];
        }
    }

    public int simulate(double[] x, boolean can_create_new_node) {
        boolean new_node = can_create_new_node;
        int C = getNodeCount();

        int winner = -1;

        if (can_create_new_node) {
            for (int i = 0; i < C; ++i) {
                activation_values.set(i,
                    choice_function(x, i));
            }

            for (int i = 0; i < C; ++i) {
                int J = template_with_max_activation_value();
                if (J == -1) break;

                double match_value = match_function(x, J);
                if (match_value > rho) {
                    update_node(x, J);
                    winner = J;
                    new_node = false;
                    break;
                } else {
                    activation_values.set(J, 0.0);
                }
            }

            if (new_node) {
                addNode(x);
                winner = getNodeCount() - 1;
            }
        } else {
            double max_match_value = Double.NEGATIVE_INFINITY;
            int J = -1;
            for (int j = 0; j < C; ++j) {
                double match_value = match_function(x, j);
                if (max_match_value < match_value) {
                    max_match_value = match_value;
                    J = j;
                }
            }
            winner = J;
        }

        return winner;
    }
}