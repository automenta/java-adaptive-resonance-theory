package com.github.chen0040.art.rl;

import java.util.Map;

/**
 * Created by chen0469 on 10/2/2015 0002.
 */
public class Tuple2<K, V> implements Map.Entry<K, V> {
    private final K key;
    private V value;

    public Tuple2(K key, V value){
        this.key = key;
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }

    public V setValue(V value) {
        return this.value = value;
    }
}