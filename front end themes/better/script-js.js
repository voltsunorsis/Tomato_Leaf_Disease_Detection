document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.innerHTML = '<div class="analysis-effect"></div>';
            imagePreview.appendChild(img);
            imagePreview.style.display = 'flex';
            
            // Show the predict button with a fade-in effect
            const predictButton = document.getElementById('predictButton');
            predictButton.style.display = 'inline-block';
            predictButton.style.opacity = 0;
            setTimeout(() => {
                predictButton.style.transition = 'opacity 0.5s ease-in-out';
                predictButton.style.opacity = 1;
            }, 10);
        }
        reader.readAsDataURL(file);
    }
});

document.getElementById('predictButton').addEventListener('click', function() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    
    if (file) {
        const formData = new FormData();
        formData.append('image', file);
        
        // Show loading state and analysis effect
        this.textContent = 'Analyzing...';
        this.disabled = true;
        document.querySelector('.analysis-effect').style.opacity = 1;
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide analysis effect
            document.querySelector('.analysis-effect').style.opacity = 0;
            
            const resultCard = document.getElementById('result');
            resultCard.innerHTML = `
                <h3>Analysis Result</h3>
                <p>Predicted class: ${data.class}</p>
                <p>Probability: ${(data.probability * 100).toFixed(2)}%</p>
            `;
            resultCard.style.display = 'block';
            
            document.getElementById('symptoms').querySelector('p').textContent = data.symptoms;
            document.getElementById('remedies').querySelector('p').textContent = data.remedies;
            document.getElementById('prevention').querySelector('p').textContent = data.prevention;
            
            // Show the info cards with a staggered fade-in effect
            const infoCards = document.querySelectorAll('.info-card');
            infoCards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.display = 'block';
                    card.style.opacity = 0;
                    card.style.transform = 'translateY(20px)';
                    card.style.transition = 'opacity 0.5s ease-in-out, transform 0.5s ease-in-out';
                    setTimeout(() => {
                        card.style.opacity = 1;
                        card.style.transform = 'translateY(0)';
                    }, 10);
                }, index * 200);
            });
            
            // Reset button state
            this.textContent = 'Analyze';
            this.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = 'An error occurred during analysis.';
            document.querySelector('.analysis-effect').style.opacity = 0;
            
            // Reset button state
            this.textContent = 'Analyze';
            this.disabled = false;
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