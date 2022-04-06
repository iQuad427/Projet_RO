package com.inquad.photomanager.combo_box;

public enum LocalisationFormat {
    CITY_COUNTRY("city_country"),
    COUNTRY_CITY("country_city");

    private final String label;

    LocalisationFormat(String label) {
        this.label = label;
    }

    public String toString() {
        return label;
    }
}
