package com.github.chen0040.art.rl.q;

/**
 * Created by chen0469 on 10/3/2015 0003.
 */
public class QValue{
    private final boolean valid;
    private final double value;

    public boolean isValid(){
        return valid;
    }

    public double getValue(){
        return value;
    }

    public QValue(double value){
        valid = true;
        this.value = value;
    }

    public QValue(){
        valid = false;
        value = 0;
    }

    public static QValue Invalid() {
        return new QValue();
    }
}