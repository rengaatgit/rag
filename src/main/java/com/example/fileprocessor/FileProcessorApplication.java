package com.example.fileprocessor;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot entry point.
 *
 * On startup:
 *   1. Spring loads all @Configuration beans (including TemporalConfig)
 *   2. TemporalConfig creates WorkflowClient + Worker
 *   3. Worker registers Workflow + Activity implementations
 *   4. Worker starts polling Temporal Server for tasks
 *   5. REST Controller is ready to accept PUT /api/files/process
 *
 * Run this class to start the application.
 * Make sure Temporal Server is running first:
 *   docker run --rm -p 7233:7233 -p 8080:8080 temporalio/auto-setup:latest
 *
 * Then access Temporal UI at: http://localhost:8080
 */
@SpringBootApplication
public class FileProcessorApplication {

    public static void main(String[] args) {
        SpringApplication.run(FileProcessorApplication.class, args);
    }
}
