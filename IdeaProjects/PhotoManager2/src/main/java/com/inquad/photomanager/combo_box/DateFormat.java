package com.inquad.photomanager.combo_box;

public enum DateFormat {
    DAY_MONTH_YEAR("dd_mm_yyyy"),
    MONTH_DAY_YEAR("mm_dd_yyyy"),
    YEAR_DAY_MONTH("yyyy_dd_mm");

    private final String label;

    DateFormat(String label) {
        this.label = label;
    }

    public String toString() {
        return label;
    }
}
