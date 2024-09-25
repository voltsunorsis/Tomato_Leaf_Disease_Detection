document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const analyzeButtonContainer = document.getElementById('analyzeButtonContainer');
    const resultContainer = document.getElementById('resultContainer');
    const predictedClass = document.getElementById('predictedClass');
    const probability = document.getElementById('probability');
    const symptoms = document.getElementById('symptoms');
    const remedies = document.getElementById('remedies');
    const prevention = document.getElementById('prevention');

    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);
                analyzeButtonContainer.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });

    analyzeButton.addEventListener('click', function() {
        const file = imageUpload.files[0];
        if (file) {
            analyzeImage(file);
        } else {
            alert('Please select an image first.');
        }
    });

    function analyzeImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        analyzeButton.textContent = 'Analyzing...';
        analyzeButton.disabled = true;

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
            analyzeButton.textContent = 'Analyze';
            analyzeButton.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
            analyzeButton.textContent = 'Analyze';
            analyzeButton.disabled = false;
        });
    }

    function displayResults(result) {
        predictedClass.textContent = `Predicted class: ${result.class}`;
        probability.textContent = `Probability: ${(result.probability * 100).toFixed(2)}%`;
        symptoms.textContent = result.symptoms;
        remedies.textContent = result.remedies;
        prevention.textContent = result.prevention;
        
        resultContainer.style.display = 'block';
    }
});