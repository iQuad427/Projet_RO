package com.inquad.photomanager.model;

public class FileMetadata {
    private String name = "no_name";
    private String type = "no_type";
    private String size = "no_size";
    private String modified = "no_modifies";
    private String timestamp = "no_timestamp";
    private String extension = "no_extension";

    public FileMetadata() {}

    public void setName(String name) {
        this.name = name;
    }

    public void setType(String type) {
        this.type = type;
    }

    public void setSize(String size) {
        this.size = size;
    }

    public void setModified(String modified) {
        this.modified = modified;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public void setExtension(String extension) {
        this.extension = extension;
    }

    public String getName() {
        return name;
    }

    public String getType() {
        return type;
    }

    public String getSize() {
        return size;
    }

    public String getModified() {
        return modified;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public String getExtension() {
        return extension;
    }
}
