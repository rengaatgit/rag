package com.example.fileprocessor.activity;

import com.example.fileprocessor.model.FileProcessResult;
import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

/**
 * ════════════════════════════════════════════════════════════════
 *  ACTIVITY 2 INTERFACE — OCR Text Extraction
 * ════════════════════════════════════════════════════════════════
 *
 * WHAT:   Declares the contract for "extract text via OCR" work unit.
 *
 * PATTERN — Takes the baton, adds its own data, passes baton forward:
 *   INPUT:  FileProcessResult (localFilePath + newFileName filled by Activity 1)
 *   OUTPUT: Same FileProcessResult with extractedText now filled in
 *
 * This activity is responsible for:
 *   1. Reading the file path from the result
 *   2. Calling an external OCR API (e.g. Tesseract REST, Google Vision, AWS Textract)
 *   3. Getting the extracted text
 *   4. Storing it in result.extractedText
 *   5. Returning the updated result
 *
 * Replace OcrActivityImpl with a different impl to switch OCR providers
 * without touching the Workflow or other activities.
 */
@ActivityInterface
public interface OcrActivity {

    @ActivityMethod
    FileProcessResult extractText(FileProcessResult result);
}
