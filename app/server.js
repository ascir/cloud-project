const express = require('express');
const logger = require('morgan');
const AWS = require("aws-sdk");
const multer = require('multer');
const path = require('path');

require('dotenv').config();

const app = express();
const port = 3000;

app.use(logger('tiny'));

AWS.config.update({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    sessionToken: process.env.AWS_SESSION_TOKEN,
    region: "ap-southeast-2",
});

console.log("ACCESS KEY", process.env.AWS_ACCESS_KEY_ID);

const s3 = new AWS.S3();
const bucketName = "tfjsgrp11";

// Multer middleware to handle file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });



// Render static page
app.use(express.static('public'));

// Endpoint for file upload
app.post('/upload', upload.single('file'), (req, res) => {
  const file = req.file;

  try {
    // Upload the file to S3 bucket
    const params = {
      Bucket: bucketName,
      Key: `${Date.now()}_${path.basename(file.originalname)}`,
      Body: file.buffer,
      ContentType: file.mimetype,
      ACL: 'public-read' // Set appropriate ACL for your use case
    };

    s3.upload(params, (err, data) => {
      if (err) {
        console.error(err);
        return res.status(500).json({ error: 'Failed to upload file to S3' });
      }

      // File uploaded successfully, return the S3 URL
      const fileUrl = data.Location;
      return res.status(200).json({ message: 'File uploaded successfully', fileUrl });
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Failed to upload file to S3' });
  }
});

app.post('/train-model', upload.single('file'), async (req, res) => {
  try {
    const { optimizer, lossFunction, epochs, targetVariable } = req.body;

    // Preprocess the uploaded CSV file
    const { features, target } = preprocessData(req.file.buffer.toString(), targetVariable);

    // Upload processed data to S3 bucket
    const params = {
      Bucket: bucketName,
      Key: `processed_data_${Date.now()}.json`,
      Body: JSON.stringify({ features, target }),
      ContentType: 'application/json',
      ACL: 'public-read'
    };

    await s3.upload(params).promise();

    // Train the model based on the number of dimensions
    const trainedModel = trainModel(features, target, optimizer, lossFunction, epochs);

    // Send response with trained model or model details
    res.status(200).json({ message: 'Model trained successfully', trainedModel });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to train the model' });
  }
}); 

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});