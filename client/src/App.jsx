import React, { useState, useRef } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import { Toaster } from "react-hot-toast";
import "./app.css";

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null); // State for storing response result
  const fileInputRef = useRef(null);

  const handleImageUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile && (uploadedFile.type === "image/jpeg" || uploadedFile.type === "image/png")) {
      setSelectedImage(URL.createObjectURL(uploadedFile));
      setFile(uploadedFile);
    } else {
      toast.error("Only .jpg and .png formats are allowed!");
    }
  };

  const handleUpload = async () => {
    setResult(null);
    if (!file) {
      toast.error("No file selected!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      if (res.data.isFace == 1) {
        if (res.data.severety_level == 0) {
          toast.success("No Acne Detected");
        }else{
          toast.success(res.data.message);
        }
        setResult(res.data);
      } else {
        toast.error(res.data.message);
      }
      fileInputRef.current.value = null;
    } catch (error) {
      toast.error(error.response?.data?.error || "Error uploading file!");
    }
  };

  const handleCancel = () => {
    setSelectedImage(null);
    setFile(null);
    setResult(null);
    fileInputRef.current.value = null;
  };

  return (
    <>
      <header className="flex flex-col justify-center bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-center shadow-lg" style={{ height: "70px" }}>
        <h1 className="text-3xl font-extrabold tracking-wide">Acne Classification</h1>
        <p className="text-sm font-light">Upload an image and get an AI-powered analysis</p>
      </header>

      <div className="min-h-screen bg-gray-100 flex justify-center p-6">
        <div
          className={`flex gap-10 justify-center w-full max-w-5xl`}
        >
          <div className="bg-white shadow-md rounded-lg p-6 flex-grow upload-container">
            <h2 className="text-xl font-semibold text-gray-800 text-center mb-4">
              Upload Your Image
            </h2>
            <div className="mb-4">
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg, .png"
                onChange={handleImageUpload}
                className="block w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>
            {selectedImage && (
              <div className="relative">
                <img
                  src={selectedImage}
                  alt="Preview"
                  className="w-full h-full rounded-lg border border-gray-300"
                />
                <div className="flex justify-between mt-4">
                  <button
                    onClick={handleUpload}
                    className="px-4 py-2 bg-blue-500 text-white font-medium rounded hover:bg-blue-600"
                  >
                    Upload
                  </button>
                  <button
                    onClick={handleCancel}
                    className="px-4 py-2 bg-red-500 text-white font-medium rounded hover:bg-red-600"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>

          {result && (
            <div className="bg-gradient-to-r from-white to-blue-50 shadow-lg rounded-lg p-6 flex-grow detail-container border border-gray-200">
              <h3 className="text-2xl text-center font-bold text-blue-600 mb-6">
                Analysis Results
              </h3>
              <div className="text-center space-y-4">
                {/* <p className="text-gray-800">
                  <span className="font-medium text-lg text-gray-900">Detected Acne Marks:</span>{" "}
                  <span className="text-gray-700 text-lg">{result.Detected_acne_marks}</span>
                </p> */}
                <p className="text-gray-800">
                  <span className="font-medium text-lg text-gray-900">Severity Level:</span>{" "}
                  <span className="text-gray-700 text-lg">{result.severety_level}</span>
                </p>
              </div>
              <div className="mt-8">
                <h4 className="font-semibold text-gray-800 text-lg text-center">Suggested Medicines:</h4>
                <ul className="list-none mt-4 space-y-2 text-gray-700 text-center">
                  {result["suggested_medicine"].map((medicine, index) => (
                    <li key={index} className="bg-blue-100 inline-block ml-4 px-4 py-2 rounded-lg text-blue-900">
                      {medicine}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

          )}
        </div>
      </div>

      <Toaster />
    </>
  );
};

export default App;
