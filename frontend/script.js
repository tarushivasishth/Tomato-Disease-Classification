document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('imageInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show loading state
    document.getElementById('result').style.display = 'none';
    document.getElementById('class').innerText = 'Class: Loading...';
    document.getElementById('confidence').innerText = 'Confidence: Loading...';

    try {
        const response = await fetch('https://tomato-disease-classification-2.onrender.com/predict', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            // Display prediction result
            document.getElementById('result').style.display = 'block';
            document.getElementById('class').innerText = `Class: ${data.class}`;
            document.getElementById('confidence').innerText = `Confidence: ${data.confidence}`;
        } else {
            throw new Error('Error uploading the image');
        }
    } catch (error) {
        console.error(error);
        alert('Error occurred while classifying the image');
    }
});
