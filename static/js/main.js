/**
 * Main JavaScript file for Exoplanet Detection AI
 * Handles client-side interactions and API calls
 */

// Global variables
let currentModel = null;
let uploadedData = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Add event listeners
    setupEventListeners();
    
    // Load initial data
    loadInitialData();
    
    // Setup drag and drop for file uploads
    setupDragAndDrop();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // File upload form
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // Model training form
    const trainForm = document.getElementById('trainForm');
    if (trainForm) {
        trainForm.addEventListener('submit', handleModelTraining);
    }
    
    // Single prediction form
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        predictForm.addEventListener('submit', handleSinglePrediction);
    }
}

/**
 * Setup drag and drop functionality
 */
function setupDragAndDrop() {
    const fileInput = document.getElementById('file');
    if (!fileInput) return;
    
    const uploadArea = fileInput.closest('.card-body');
    if (!uploadArea) return;
    
    // Create drag and drop area
    const dragArea = document.createElement('div');
    dragArea.className = 'file-upload-area mb-3';
    dragArea.innerHTML = `
        <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
        <h5>Drag and drop your file here</h5>
        <p class="text-muted">or click to browse</p>
    `;
    
    // Insert before file input
    fileInput.parentNode.insertBefore(dragArea, fileInput);
    fileInput.style.display = 'none';
    
    // Drag and drop events
    dragArea.addEventListener('dragover', handleDragOver);
    dragArea.addEventListener('dragleave', handleDragLeave);
    dragArea.addEventListener('drop', handleDrop);
    dragArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

/**
 * Handle drop event
 */
function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('file').files = files;
        handleFileSelect({ target: { files: files } });
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        const dragArea = document.querySelector('.file-upload-area');
        if (dragArea) {
            dragArea.innerHTML = `
                <i class="fas fa-file-csv fa-3x text-success mb-3"></i>
                <h5>${file.name}</h5>
                <p class="text-muted">${formatFileSize(file.size)}</p>
                <button type="button" class="btn btn-sm btn-outline-secondary" onclick="clearFile()">
                    <i class="fas fa-times"></i> Remove
                </button>
            `;
        }
    }
}

/**
 * Clear selected file
 */
function clearFile() {
    document.getElementById('file').value = '';
    const dragArea = document.querySelector('.file-upload-area');
    if (dragArea) {
        dragArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
            <h5>Drag and drop your file here</h5>
            <p class="text-muted">or click to browse</p>
        `;
    }
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Handle file upload
 */
function handleFileUpload(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    submitBtn.disabled = true;
    
    // Submit form normally (let Flask handle it)
    e.target.submit();
}

/**
 * Handle model training
 */
function handleModelTraining(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Show progress
    showTrainingProgress();
    
    // Send request
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideTrainingProgress();
        if (result.success) {
            showTrainingResults(result);
        } else {
            showError('Training failed: ' + result.error);
        }
    })
    .catch(error => {
        hideTrainingProgress();
        showError('Training failed: ' + error.message);
    });
}

/**
 * Handle single prediction
 */
function handleSinglePrediction(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const features = {};
    
    // Extract feature values
    for (const [key, value] of formData.entries()) {
        if (key !== 'csrf_token') {
            features[key] = parseFloat(value) || 0;
        }
    }
    
    // Send prediction request
    fetch('/predict_single', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            showPredictionResult(result.result);
        } else {
            showError('Prediction failed: ' + result.error);
        }
    })
    .catch(error => {
        showError('Prediction failed: ' + error.message);
    });
}

/**
 * Show training progress
 */
function showTrainingProgress() {
    const progressDiv = document.getElementById('trainingProgress');
    if (progressDiv) {
        progressDiv.style.display = 'block';
    }
    
    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 95) progress = 95;
        
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        if (progressText) {
            if (progress < 30) {
                progressText.textContent = 'Loading dataset...';
            } else if (progress < 60) {
                progressText.textContent = 'Training model...';
            } else if (progress < 90) {
                progressText.textContent = 'Validating model...';
            } else {
                progressText.textContent = 'Finalizing...';
            }
        }
    }, 1000);
    
    // Store interval ID for cleanup
    window.trainingInterval = interval;
}

/**
 * Hide training progress
 */
function hideTrainingProgress() {
    if (window.trainingInterval) {
        clearInterval(window.trainingInterval);
        window.trainingInterval = null;
    }
    
    const progressDiv = document.getElementById('trainingProgress');
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
    
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = '100%';
    }
}

/**
 * Show training results
 */
function showTrainingResults(result) {
    const resultsDiv = document.getElementById('trainingResults');
    const contentDiv = document.getElementById('resultsContent');
    
    if (!resultsDiv || !contentDiv) return;
    
    const trainingResults = result.training_results;
    
    contentDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6 class="card-title">Model Performance</h6>
                        <div class="mb-2">
                            <strong>Accuracy:</strong> 
                            <span class="badge bg-success fs-6">${(trainingResults.accuracy * 100).toFixed(2)}%</span>
                        </div>
                        <div class="mb-2">
                            <strong>CV Score:</strong> 
                            <span class="badge bg-info fs-6">${(trainingResults.cv_mean * 100).toFixed(2)}% ± ${(trainingResults.cv_std * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-light">
                    <div class="card-body">
                        <h6 class="card-title">Confusion Matrix</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <tbody>
                                    ${trainingResults.confusion_matrix.map((row, i) => `
                                        <tr>
                                            ${row.map((cell, j) => `<td class="text-center">${cell}</td>`).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        ${trainingResults.feature_importance ? `
            <div class="mt-4">
                <h6>Top 10 Most Important Features</h6>
                <div class="row">
                    ${Object.entries(trainingResults.feature_importance)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([feature, importance]) => `
                            <div class="col-md-6 mb-2">
                                <div class="d-flex justify-content-between align-items-center">
                                    <span class="small">${feature}</span>
                                    <span class="badge bg-secondary">${(importance * 100).toFixed(1)}%</span>
                                </div>
                                <div class="progress" style="height: 6px;">
                                    <div class="progress-bar bg-primary" style="width: ${importance * 100}%"></div>
                                </div>
                            </div>
                        `).join('')}
                </div>
            </div>
        ` : ''}
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Show prediction result
 */
function showPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    if (!resultDiv) return;
    
    const confidenceColor = result.confidence > 0.8 ? 'success' : 
                           result.confidence > 0.6 ? 'warning' : 'danger';
    
    resultDiv.innerHTML = `
        <div class="alert alert-${confidenceColor}">
            <h5 class="alert-heading">
                <i class="fas fa-star"></i> Prediction Result
            </h5>
            <hr>
            <div class="row">
                <div class="col-md-6">
                    <strong>Predicted Class:</strong><br>
                    <span class="badge bg-primary fs-6">${result.prediction}</span>
                </div>
                <div class="col-md-6">
                    <strong>Confidence:</strong><br>
                    <span class="badge bg-${confidenceColor} fs-6">${(result.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
            <hr>
            <h6>Class Probabilities:</h6>
            <div class="row">
                ${Object.entries(result.class_probabilities).map(([className, prob]) => `
                    <div class="col-md-4 mb-2">
                        <div class="d-flex justify-content-between">
                            <span class="small">${className}</span>
                            <span class="badge bg-secondary">${(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div class="progress" style="height: 4px;">
                            <div class="progress-bar bg-secondary" style="width: ${prob * 100}%"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

/**
 * Show error message
 */
function showError(message) {
    // Create or update error alert
    let errorDiv = document.getElementById('errorAlert');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'errorAlert';
        errorDiv.className = 'alert alert-danger alert-dismissible fade show';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span id="errorMessage"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of main content
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(errorDiv, main.firstChild);
        }
    }
    
    document.getElementById('errorMessage').textContent = message;
    errorDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorDiv) {
            errorDiv.style.display = 'none';
        }
    }, 5000);
}

/**
 * Show success message
 */
function showSuccess(message) {
    // Create or update success alert
    let successDiv = document.getElementById('successAlert');
    if (!successDiv) {
        successDiv = document.createElement('div');
        successDiv.id = 'successAlert';
        successDiv.className = 'alert alert-success alert-dismissible fade show';
        successDiv.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span id="successMessage"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of main content
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(successDiv, main.firstChild);
        }
    }
    
    document.getElementById('successMessage').textContent = message;
    successDiv.style.display = 'block';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        if (successDiv) {
            successDiv.style.display = 'none';
        }
    }, 3000);
}

/**
 * Load initial data
 */
function loadInitialData() {
    // Load model info
    loadModelInfo();
    
    // Load dataset info
    loadDatasetInfo();
}

/**
 * Load model information
 */
function loadModelInfo() {
    fetch('/model_info')
        .then(response => response.json())
        .then(data => {
            updateModelStatus(data);
        })
        .catch(error => {
            console.error('Error loading model info:', error);
        });
}

/**
 * Load dataset information
 */
function loadDatasetInfo() {
    fetch('/datasets')
        .then(response => response.json())
        .then(data => {
            updateDatasetInfo(data);
        })
        .catch(error => {
            console.error('Error loading dataset info:', error);
        });
}

/**
 * Update model status display
 */
function updateModelStatus(data) {
    const statusElement = document.getElementById('model-status');
    const statusContent = document.getElementById('model-status-content');
    
    if (statusElement) {
        if (data.is_trained) {
            statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Model Ready';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> No Model';
        }
    }
    
    if (statusContent) {
        if (data.is_trained) {
            statusContent.innerHTML = `
                <div class="row">
                    <div class="col-6">
                        <strong>Model Type:</strong><br>
                        <span class="badge bg-primary">${data.model_type}</span>
                    </div>
                    <div class="col-6">
                        <strong>Features:</strong><br>
                        <span class="badge bg-info">${data.n_features}</span>
                    </div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-12">
                        <strong>Classes:</strong><br>
                        ${data.classes.map(cls => `<span class="badge bg-secondary me-1">${cls}</span>`).join('')}
                    </div>
                </div>
            `;
        } else {
            statusContent.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    No model is currently trained. Please train a model first.
                </div>
            `;
        }
    }
}

/**
 * Update dataset information display
 */
function updateDatasetInfo(data) {
    const content = document.getElementById('dataset-info');
    if (!content) return;
    
    let html = '';
    
    for (const [name, info] of Object.entries(data)) {
        html += `
            <div class="mb-3">
                <h6 class="text-capitalize">${name} Dataset</h6>
                <p class="small mb-1">
                    <strong>Records:</strong> ${info.shape[0].toLocaleString()}<br>
                    <strong>Features:</strong> ${info.shape[1]}
                </p>
            </div>
        `;
    }
    
    if (html === '') {
        html = '<p class="text-muted small">No datasets loaded. Please place CSV files in the data/ directory.</p>';
    }
    
    content.innerHTML = html;
}

/**
 * Utility function to format numbers
 */
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

/**
 * Utility function to create loading spinner
 */
function createSpinner(size = 'sm') {
    return `<div class="spinner-border spinner-border-${size} text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>`;
}

// Export functions for global access
window.clearFile = clearFile;
window.loadModelInfo = loadModelInfo;
window.loadDatasetInfo = loadDatasetInfo;
