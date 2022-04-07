package com.github.chen0040.art;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ResonantAdapter1 {

    /** choice parameter */
    protected double alpha;

    /** base resonance threshold */
    protected double rho;

    /** learning rate */
    protected double beta;

    protected List<double[]> weights;

    /** activation values */
    protected List<Double> activation;

    protected int inputCount;

    public ResonantAdapter1() {
        this(0, 0);
    }

    public ResonantAdapter1(int inputCount, int initialNeuronCount) {
        this.inputCount = inputCount;

        weights = new ArrayList<>(/*?*/);
        activation = new ArrayList<>(/*?*/);

        for (int i = 0; i < initialNeuronCount; ++i)
            addNode();
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
        addNodeDirect(neuron);
    }
    public void addNode(double[] x) {
        addNodeDirect(x.clone());
    }

    public final void addNodeDirect(double[] x) {
        weights.add(x);
        activation.add(0.0);
    }

    /** choice function */
    protected double choose(double[] x, int j) {
        double[] Wj = weights.get(j);
        double a = 0, b = 0;
        for (int i = 0; i < x.length; ++i) {
            double Wji = Wj[i];
            a += Math.abs(x[i] * Wji); // norm1
            b += Math.abs(Wji); //  norm1
        }
        return a / (alpha + b);
    }

    /** template with max activation */
    protected int templateActive() {
        int C = nodeCount();
        double vMax = Double.NEGATIVE_INFINITY;
        int t = -1;
        for (int i = 0; i < C; ++i) {
            double v = activation.get(i);
            if (v > vMax) {
                vMax = v;
                t = i;
            }
        }
        return t;
    }

    public int nodeCount() {
        return weights.size();
    }

    protected double match(double[] x, int j) {
        double a = 0, b = 0;
        double[] W_j = weights.get(j);
        for (int i = 0; i < x.length; ++i) {
            double Xi = x[i];
            a += Math.abs(Xi * W_j[i]); // norm1
            b += Math.abs(Xi); //  norm1
        }
        return a / b;
    }

    protected void updateNode(double[] x, int j) {
        double[] W_j = weights.get(j);
        for (int i = 0; i < x.length; ++i) {
            double Wji = W_j[i];
            W_j[i] = (1 - beta) * Wji + beta * Wji * x[i];
        }
    }

    public int simulate(double[] x, boolean can_create_new_node) {
        int C = nodeCount();
        return can_create_new_node ?
                simulateNew(x, C) :
                simulateExisting(x, C);
    }

    private int simulateExisting(double[] x, int n) {
        double winnerValue = Double.NEGATIVE_INFINITY;
        int winner = -1;
        for (int j = 0; j < n; ++j) {
            double value = match(x, j);
            if (winnerValue < value) {
                winnerValue = value;
                winner = j;
            }
        }
        return winner;
    }

    private int simulateNew(double[] x, int n) {
        for (int i = 0; i < n; ++i)
            activation.set(i, choose(x, i));

        int winner = -1;
        boolean new_node = true;
        for (int i = 0; i < n; ++i) {
            int J = templateActive();
            if (J == -1) break;

            double match_value = match(x, J);

            //TODO if multiple == rho, random choose (rare)
            if (match_value > rho) {
                updateNode(x, J);
                winner = J;
                new_node = false;
                break;
            } else {
                activation.set(J, 0.0);
            }
        }

        if (new_node) {
            addNode(x);
            winner = nodeCount() - 1;
        }
        return winner;
    }
}