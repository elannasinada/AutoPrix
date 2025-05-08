document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const form = document.getElementById('prediction-form');
    const makeSelect = document.getElementById('make');
    const modelSelect = document.getElementById('model');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsContent = document.getElementById('results-content');
    const errorMessage = document.getElementById('error-message');
    const linearPrice = document.getElementById('linear-price');
    const lassoPrice = document.getElementById('lasso-price');
    const xgboostPrice = document.getElementById('xgboost-price');
    const averagePrice = document.getElementById('average-price');
    const resetBtn = document.getElementById('reset-btn');
    const errorResetBtn = document.getElementById('error-reset-btn');

    // Progress navigation
    const nextButtons = document.querySelectorAll('.next-btn');
    const backButtons = document.querySelectorAll('.back-btn');
    const progressSteps = document.querySelectorAll('.progress-step');
    const formSteps = document.querySelectorAll('.form-step');

    let currentStep = 0;

    // Initialize
    updateCarModels(makeSelect.value);

    makeSelect.addEventListener('change', function () {
        updateCarModels(this.value);
    });

    // Validation des étapes
    nextButtons.forEach(button => {
        button.addEventListener('click', () => {
            const step = parseInt(button.closest('.form-step').dataset.step);
            if (validateStep(step)) {
                goToStep(step + 1);
            }
        });
    });

    backButtons.forEach(button => {
        button.addEventListener('click', () => {
            const step = parseInt(button.closest('.form-step').dataset.step);
            goToStep(step - 1);
        });
    });

    // Interception de la soumission du formulaire
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        // Validation finale sur la dernière étape visible
        if (!validateStep(currentStep + 1)) return;

        // Form HTML5 validation (au cas où des champs invisibles sont invalides)
        if (!form.reportValidity()) return;

        submitForm();
    });

    resetBtn.addEventListener('click', resetForm);
    errorResetBtn.addEventListener('click', resetForm);

    function updateCarModels(make) {
        if (!make) return;

        modelSelect.innerHTML = '<option value="" selected disabled>Chargement...</option>';

        fetch(`/car-models/${make}`)
            .then(response => response.json())
            .then(models => {
                modelSelect.innerHTML = '<option value="" selected disabled>Sélectionnez un modèle</option>';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Erreur lors du chargement des modèles:', error);
                modelSelect.innerHTML = '<option value="" selected disabled>Erreur</option>';
            });
    }

    function validateStep(step) {
        const stepElement = document.querySelector(`.form-step[data-step="${step}"]`);
        const inputs = stepElement.querySelectorAll('input, select');
        let isValid = true;

        inputs.forEach(input => {
            if (!input.checkValidity()) {
                input.classList.add('invalid');
                input.reportValidity();
                isValid = false;
            } else {
                input.classList.remove('invalid');
            }
        });

        return isValid;
    }

    function goToStep(step) {
        currentStep = step - 1;

        formSteps.forEach(formStep => formStep.classList.remove('active'));
        progressSteps.forEach((progressStep, index) => {
            progressStep.classList.toggle('completed', index < step - 1);
            progressStep.classList.toggle('active', index === step - 1);
        });

        const target = document.querySelector(`.form-step[data-step="${step}"]`);
        if (target) target.classList.add('active');
    }

    function submitForm() {
        resultsContainer.classList.remove('hidden');
        loadingIndicator.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        errorMessage.classList.add('hidden');

        resultsSection.scrollIntoView({ behavior: 'smooth' });

        const formData = new FormData(form);
        // Debug: Log form data before submission
        console.log("Form Data:");
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    setTimeout(() => {
                        loadingIndicator.classList.add('hidden');
                        resultsContent.classList.remove('hidden');
                        animatePrice(linearPrice, data.linear_prediction);
                        animatePrice(lassoPrice, data.lasso_prediction);
                        animatePrice(xgboostPrice, data.xgboost_prediction);
                        animatePrice(averagePrice, data.average_prediction);
                    }, 1500);
                } else {
                    throw new Error(data.error || 'Une erreur inconnue est survenue');
                }
            })
            .catch(error => {
                console.error('Erreur :', error);
                loadingIndicator.classList.add('hidden');
                errorMessage.classList.remove('hidden');
            });
    }

    function animatePrice(element, finalPrice) {
        const numericPrice = parseInt(finalPrice.replace(/[^0-9]/g, '')) || 0;
        let startPrice = 0;
        const duration = 1500;
        const interval = 20;
        const steps = duration / interval;
        const increment = numericPrice / steps;

        const timer = setInterval(() => {
            startPrice += increment;
            if (startPrice >= numericPrice) {
                clearInterval(timer);
                element.textContent = numericPrice.toLocaleString() + ' DH';
            } else {
                element.textContent = Math.floor(startPrice).toLocaleString() + ' DH';
            }
        }, interval);
    }

    function resetForm() {
        resultsContainer.classList.add('hidden');
        form.reset();
        modelSelect.innerHTML = '<option value="" selected disabled>Sélectionnez une marque d\'abord</option>';
        goToStep(1);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});