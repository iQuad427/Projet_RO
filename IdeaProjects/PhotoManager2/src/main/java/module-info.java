module com.inquad.photomanager {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.desktop;
    requires org.apache.commons.imaging;
    requires metadata.extractor;
    requires org.apache.poi.poi;
    requires org.apache.poi.ooxml;

    opens com.inquad.photomanager to javafx.fxml;
    exports com.inquad.photomanager;
    exports com.inquad.photomanager.combo_box;
    opens com.inquad.photomanager.combo_box to javafx.fxml;
    exports com.inquad.photomanager.views;
    opens com.inquad.photomanager.views to javafx.fxml;
}