package com.inquad.photomanager.combo_box;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Format {
    private static List<RenameFormat> format = Collections.emptyList();

    private Format() {}

    public static List<RenameFormat> getFormat() {
        if (format.isEmpty()) {
            format = new ArrayList<>();
        }
        return format;
    }
}
