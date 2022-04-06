package com.inquad.photomanager.combo_box;

public enum RenameFormat {
    LOCALISATION("localisation"),
    IDENTIFIER("identifier"),
    NAME("name"),
    DATE("date"),
    CAMERA("type"),
    SIZE("size"),
    EXTENSION("extension"),
    NONE("nothing");

    private final String label;

    RenameFormat(String label) {
        this.label = label;
    }

    public String toString() {
        return label;
    }
}
