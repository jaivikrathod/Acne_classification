import React, { useState,useRef } from "react";
import axios from "axios";
import toast from "react-hot-toast"; // Import react-hot-toast
import { Toaster } from "react-hot-toast";

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);  

  // Handle file input change
  const handleImageUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile && (uploadedFile.type === "image/jpeg" || uploadedFile.type === "image/png")) {
      setSelectedImage(URL.createObjectURL(uploadedFile));
      setFile(uploadedFile);
    } else {
      toast.error("Only .jpg and .png formats are allowed!");
    }
  };

  // Handle file upload to backend
  const handleUpload = async () => {
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
      toast.success(res.data.message); // "success"
      setSelectedImage(null); // Clear the preview
      setFile(null);
      fileInputRef.current.value = null;
    } catch (error) {
      toast.error(error.response?.data?.error || "Error uploading file!");
    }
  };

  // Handle cancel
  const handleCancel = () => {
    setSelectedImage(null);
    setFile(null);
    fileInputRef.current.value = null;
  };

  return (
    <>
      <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">

        <div className="bg-white shadow-md rounded-lg p-6 max-w-md w-full">
          <h2 className="text-xl font-semibold text-gray-800 text-center mb-4">Upload Your Image</h2>
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
                className="w-full h-auto rounded-lg mb-4 border border-gray-300"
              />
              <div className="flex justify-between">
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
      </div>
        <Toaster />
    </>

  );
}

export default App;
