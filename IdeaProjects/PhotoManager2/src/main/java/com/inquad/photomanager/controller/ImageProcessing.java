package com.inquad.photomanager.controller;

import com.drew.imaging.ImageMetadataReader;
import com.drew.imaging.ImageProcessingException;
import com.drew.metadata.Directory;
import com.drew.metadata.Metadata;
import com.drew.metadata.Tag;
import com.inquad.photomanager.combo_box.RenameFormat;
import com.inquad.photomanager.model.FileMetadata;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.CreationHelper;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class ImageProcessing {

    public void renameFiles(File directory, List<RenameFormat> renameFormats) throws IOException, ImageProcessingException {
        File[] fileList = directory.listFiles();

        if (fileList != null) {
            int identifier = 0;
            for (File file : fileList) {
                if (isImage(file)) {
                    FileMetadata fileMetadata = extractMetadata(file);
                    StringBuilder newName = new StringBuilder();

                    identifier++;
                    boolean needToDelete = false;
                    int renameCount = 0;
                    for (RenameFormat renameFormat : renameFormats) {
                        if (renameCount != 0) {
                            newName.append("_");
                            needToDelete = true;
                        }
                        switch (renameFormat.toString()) {
                            case "name" -> {
                                newName.append(fileMetadata.getName().split("\\.")[0]);
                                renameCount++;
                            }
                            case "identifier" -> {
                                newName.append(String.format("%06d", identifier));
                                renameCount++;
                            }
                            case "date" -> {
                                if (!Objects.equals(fileMetadata.getTimestamp(), "no_timestamp")) {
                                    newName.append(fileMetadata.getTimestamp().split(" ")[0].replace(":", "-"));
                                    newName.append("--");
                                    newName.append(fileMetadata.getTimestamp().split(" ")[1].replace(":", "-"));

                                } else {
                                    newName.append("no_date");
                                }
                                renameCount++;
                            }
                            case "type" -> {
                                newName.append(fileMetadata.getType().split("/")[1]);
                                renameCount++;
                            }
                            case "extension" -> {
                                newName.append(fileMetadata.getExtension());
                                renameCount++;
                            }
                            case "size" -> {
                                newName.append(fileMetadata.getSize().replace(" ", "_"));
                                renameCount++;
                            }
                            case "nothing" -> {
                                if (needToDelete) {
                                    newName.deleteCharAt(newName.length() - 1);
                                }
                            }
                        }
                    }

                    newName.append(".").append(fileMetadata.getExtension());
                    Files.copy(Paths.get(file.getPath()), Paths.get(file.getParent() + "/" + newName));
                }
            }
        }
    }

    public void exportMetadata(File directory) throws IOException, ImageProcessingException {
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Images Metadata");

        writeHeaderLine(sheet);

        File[] fileList = directory.listFiles();
        List<FileMetadata> allMetadata = new ArrayList<>();

        if (fileList != null) {
            for (File file : fileList) {
                if (isImage(file)) {
                    FileMetadata fileMetadata = extractMetadata(file);
                    allMetadata.add(fileMetadata);
                }
            }
        }

        writeDataLines(allMetadata, workbook, sheet);

        FileOutputStream outputStream = new FileOutputStream(directory.getPath() + "/metadata.xlsx");
        workbook.write(outputStream);
        workbook.close();
    }

    public FileMetadata extractMetadata(File file) throws ImageProcessingException, IOException {
        FileMetadata fileMetadata = new FileMetadata();

        if (isImage(file)) {
            Metadata metadata = ImageMetadataReader.readMetadata(file);

            for (Directory metaDirectory : metadata.getDirectories()) {
                List<Tag> tagList = metaDirectory.getTags().stream().toList();

                for (Tag tag : tagList) {
                    System.out.println(tag);
                    switch (tag.getTagName()) {
                        case "File Name":
                            System.out.println("Name : " + tag.getDescription());
                            fileMetadata.setName(tag.getDescription());
                            break;
                        case "Detected MIME Type":
                            System.out.println("Type : " + tag.getDescription());
                            fileMetadata.setType(tag.getDescription());
                            break;
                        case "File Size":
                            System.out.println("Size : " + tag.getDescription());
                            fileMetadata.setSize(tag.getDescription());
                            break;
                        case "File Modified Date":
                            System.out.println("Modified : " + tag.getDescription());
                            fileMetadata.setModified(tag.getDescription());
                            break;
                        case "Date/Time":
                            System.out.println("Time : " + tag.getDescription());
                            fileMetadata.setTimestamp(tag.getDescription());
                            break;
                        case "Expected File Name Extension":
                            System.out.println("Extension : " + tag.getDescription());
                            fileMetadata.setExtension(tag.getDescription());
                            break;
                    }
                }
            }
            System.out.println();
        }

        return fileMetadata;
    }

    private void writeHeaderLine(XSSFSheet sheet) {

        Row headerRow = sheet.createRow(0);

        Cell headerCell = headerRow.createCell(0);
        headerCell.setCellValue("File Name");

        headerCell = headerRow.createCell(1);
        headerCell.setCellValue("File Type");

        headerCell = headerRow.createCell(2);
        headerCell.setCellValue("File Size");

        headerCell = headerRow.createCell(3);
        headerCell.setCellValue("Modified");

        headerCell = headerRow.createCell(4);
        headerCell.setCellValue("Timestamp");
    }

    private void writeDataLines(List<FileMetadata> allMetadata, XSSFWorkbook workbook, XSSFSheet sheet) {
        Row row;
        Cell cell;

        int rowCount = 1;
        for (FileMetadata fileMetadata : allMetadata) {
            int columnCount = 0;
            row = sheet.createRow(rowCount++);

            cell = row.createCell(columnCount++);
            cell.setCellValue(fileMetadata.getName());

            cell = row.createCell(columnCount++);
            cell.setCellValue(fileMetadata.getType());

            cell = row.createCell(columnCount++);
            cell.setCellValue(fileMetadata.getSize());

            cell = row.createCell(columnCount++);
            cell.setCellValue(fileMetadata.getModified());

            cell = row.createCell(columnCount);
            cell.setCellValue(fileMetadata.getTimestamp());
        }
    }

    public void findDuplicates(File directory) throws IOException {
        File[] fileList = directory.listFiles();
        String newDirectoryPath = directory.getPath() + "/duplicates/";
        int j = 0;
        new File(newDirectoryPath).mkdirs();
        if (fileList != null) {
            for (File bufferFile : fileList) {
                if (isImage(bufferFile)) {
                    BufferedImage bufferedImage = ImageIO.read(bufferFile);
                    if (isImage(bufferFile)) {
                        for (int i = j; i < fileList.length; i++) {
                            File testFile = fileList[i];
                            if (isImage(testFile) && testFile != bufferFile) {
                                try {
                                    System.out.println("img1 : " + bufferFile.getName());
                                    System.out.println("img2 : " + testFile.getName());
                                    double percent = getDifferencePercent(bufferedImage, ImageIO.read(testFile));
                                    System.out.println("percent : " + percent);
                                    if (percent < 0.1) {
                                        Files.copy(Paths.get(bufferFile.getPath()), Paths.get(newDirectoryPath + bufferFile.getName()));
//                                        Files.move(Paths.get(bufferFile.getPath()), Paths.get(newDirectoryPath + bufferFile.getName()));
                                        System.out.println(bufferFile.getPath() + " moved to " + newDirectoryPath);
                                    }
                                } catch (IllegalArgumentException e) {
                                    System.out.println("Problem");
                                }
                            }
                        }
                    }
                }
                j++;
            }
        }
    }

    private static double getDifferencePercent(BufferedImage img1, BufferedImage img2) {
        int width = img1.getWidth();
        int height = img1.getHeight();
        int width2 = img2.getWidth();
        int height2 = img2.getHeight();
        if (width != width2 || height != height2) {
            throw new IllegalArgumentException(String.format("Images must have the same dimensions: (%d,%d) vs. (%d,%d)", width, height, width2, height2));
        }

        long diff = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                diff += pixelDiff(img1.getRGB(x, y), img2.getRGB(x, y));
            }
        }
        long maxDiff = 3L * 255 * width * height;

        return 100.0 * diff / maxDiff;
    }

    private static int pixelDiff(int rgb1, int rgb2) {
//        System.out.println("pixelDiff");
        int r1 = (rgb1 >> 16) & 0xff;
        int g1 = (rgb1 >>  8) & 0xff;
        int b1 =  rgb1        & 0xff;
        int r2 = (rgb2 >> 16) & 0xff;
        int g2 = (rgb2 >>  8) & 0xff;
        int b2 =  rgb2        & 0xff;
        return Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
    }

    public boolean isImage(File file) throws IOException {
        String mimetype = Files.probeContentType(file.toPath());
        return mimetype != null && mimetype.split("/")[0].equals("image");
    }
}

