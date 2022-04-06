package com.inquad.photomanager.views;

import com.drew.imaging.ImageProcessingException;
import com.inquad.photomanager.combo_box.DateFormat;
import com.inquad.photomanager.combo_box.LocalisationFormat;
import com.inquad.photomanager.combo_box.RenameFormat;
import com.inquad.photomanager.controller.ImageProcessing;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.stage.DirectoryChooser;
import org.apache.commons.imaging.ImageReadException;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ResourceBundle;

public class MainController implements Initializable {

    @FXML private TextField pathTextField;

    @FXML private Button executeButton;
    @FXML private Button choseButton;

    @FXML private ComboBox<RenameFormat> formatOne;
    @FXML private ComboBox<RenameFormat> formatSecond;
    @FXML private ComboBox<RenameFormat> formatThird;
    @FXML private ComboBox<RenameFormat> formatFourth;

    @FXML private ComboBox<DateFormat> dateFormatComboBox;
    @FXML private ComboBox<LocalisationFormat> localisationFormatComboBox;

    @FXML private CheckBox duplicatesCheckBox;
    @FXML private CheckBox renameCheckBox;
    @FXML private CheckBox metadataCheckBox;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        dateFormatComboBox.getItems().setAll(DateFormat.values());
        localisationFormatComboBox.getItems().setAll(LocalisationFormat.values());
        formatOne.getItems().setAll(RenameFormat.values());
        formatSecond.getItems().setAll(RenameFormat.values());
        formatThird.getItems().setAll(RenameFormat.values());
        formatFourth.getItems().setAll(RenameFormat.values());
    }

    @FXML
    protected void handleChosePathButton(ActionEvent e) {
        try {
            DirectoryChooser dir_chooser = new DirectoryChooser();
            File directory = dir_chooser.showDialog(null);
            if (directory != null) {
                pathTextField.setText(directory.getAbsolutePath());
            }
        }
        catch (Exception exc) {
            System.out.println(exc.getMessage());
        }
    }

    @FXML
    protected void handleExecuteButton(ActionEvent e) throws IOException, ImageProcessingException, ImageReadException {
        String url = pathTextField.getText();
        File directory = new File(url);
        if (directory.isDirectory()) {
            ImageProcessing processor = new ImageProcessing();
            if (renameCheckBox.isSelected()) {
                processor.renameFiles(directory, getFormat());
            }
            if (duplicatesCheckBox.isSelected()) {
                processor.findDuplicates(directory);
            }
            if (metadataCheckBox.isSelected()) {
                processor.exportMetadata(directory);
            }
        } else {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Error");
            alert.setHeaderText("An error occurred when accessing the given directory");
            alert.setContentText("Directory Path must refer to a valid directory");
            alert.showAndWait();
        }
    }

    public List<RenameFormat> getFormat() {
        List<RenameFormat> format = new ArrayList<>();

        format.add(formatOne.getValue());
        format.add(formatSecond.getValue());
        format.add(formatThird.getValue());
        format.add(formatFourth.getValue());

        return format;
    }
}
