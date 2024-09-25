document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const predictButton = document.getElementById('predictButton');
    const resultCard = document.getElementById('result');

    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.innerHTML = '<div class="analysis-effect"></div>';
                imagePreview.appendChild(img);
                imagePreview.style.display = 'flex';
                
                predictButton.style.display = 'inline-block';
                predictButton.style.opacity = 1;
            }
            reader.readAsDataURL(file);
        }
    });

    predictButton.addEventListener('click', function() {
        const file = imageUpload.files[0];
        
        if (file) {
            const formData = new FormData();
            formData.append('image', file);
            
            predictButton.textContent = 'Analyzing...';
            predictButton.disabled = true;
            document.querySelector('.analysis-effect').style.opacity = 1;
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.querySelector('.analysis-effect').style.opacity = 0;
                
                resultCard.innerHTML = `
                    <h3>Analysis Result</h3>
                    <p>Predicted class: ${data.class}</p>
                    <p>Probability: ${(data.probability * 100).toFixed(2)}%</p>
                `;
                resultCard.style.display = 'block';
                
                document.getElementById('symptoms').innerHTML = `
                    <div class="card-header">
                        <h3>Symptoms</h3>
                        <div class="icon-symptoms"></div>
                    </div>
                    <p>${data.symptoms}</p>
                `;
                document.getElementById('remedies').innerHTML = `
                    <div class="card-header">
                        <h3>Remedies</h3>
                        <div class="icon-remedies"></div>
                    </div>
                    <p>${data.remedies}</p>
                `;
                document.getElementById('prevention').innerHTML = `
                    <div class="card-header">
                        <h3>Prevention</h3>
                        <div class="icon-prevention"></div>
                    </div>
                    <p>${data.prevention}</p>
                `;
                
                const infoCards = document.querySelectorAll('.info-card');
                infoCards.forEach((card, index) => {
                    card.style.display = 'block';
                    card.style.opacity = 1;
                    card.style.transform = 'translateY(0)';
                });
                
                predictButton.textContent = 'Analyze';
                predictButton.disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                resultCard.innerHTML = 'An error occurred during analysis.';
                document.querySelector('.analysis-effect').style.opacity = 0;
                
                predictButton.textContent = 'Analyze';
                predictButton.disabled = false;
            });
        } else {
            alert('Please select an image first.');
        }
    });

    // Add hover effect to info cards
    document.querySelectorAll('.info-card').forEach(card => {
        card.addEventListener('mouseover', () => {
            card.style.transform = 'translateY(-5px)';
            card.style.boxShadow = '0 15px 30px rgba(0, 0, 0, 0.3)';
        });
        card.addEventListener('mouseout', () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.2)';
        });
    });
});