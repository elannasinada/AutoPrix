document.addEventListener('DOMContentLoaded', function() {
    // Add initial animations to elements
    animateOnScroll();
    
    // Listen for scroll events to trigger animations
    window.addEventListener('scroll', animateOnScroll);
    
    // Function to handle scroll-based animations
    function animateOnScroll() {
        const animatedElements = document.querySelectorAll('.form-container, .results-container, .price-card');
        
        animatedElements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight;
            
            // If element is in viewport
            if (elementPosition < screenPosition) {
                if (!element.classList.contains('animated')) {
                    element.classList.add('animated');
                    element.style.animation = 'fadeIn 0.6s ease-out forwards';
                }
            }
        });
    }
    
    // Add hover effects to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
    
    // Add ripple effect to buttons
    buttons.forEach(button => {
        button.addEventListener('click', createRippleEffect);
    });
    
    function createRippleEffect(event) {
        const button = event.currentTarget;
        
        const ripple = document.createElement('span');
        ripple.classList.add('ripple');
        
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;
        
        ripple.style.width = ripple.style.height = `${diameter}px`;
        ripple.style.left = `${event.clientX - (button.getBoundingClientRect().left + radius)}px`;
        ripple.style.top = `${event.clientY - (button.getBoundingClientRect().top + radius)}px`;
        
        // Remove existing ripples
        const existingRipple = button.querySelector('.ripple');
        if (existingRipple) {
            existingRipple.remove();
        }
        
        button.appendChild(ripple);
        
        // Remove ripple after animation completes
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }
    
    // Add CSS for ripple effect
    const style = document.createElement('style');
    style.textContent = `
        button {
            position: relative;
            overflow: hidden;
        }
        
        .ripple {
            position: absolute;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }
        
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
    
    // Add subtle background animation
    const container = document.querySelector('.container');
    createBackgroundAnimation(container);
    
    function createBackgroundAnimation(element) {
        // Create decoration elements
        for (let i = 0; i < 5; i++) {
            const decoration = document.createElement('div');
            decoration.classList.add('bg-decoration');
            
            // Randomize position and size
            const size = Math.random() * 100 + 50;
            decoration.style.width = `${size}px`;
            decoration.style.height = `${size}px`;
            decoration.style.left = `${Math.random() * 100}%`;
            decoration.style.top = `${Math.random() * 100}%`;
            decoration.style.opacity = `${Math.random() * 0.1}`;
            
            // Add to container
            element.appendChild(decoration);
            
            // Animate with random duration
            animation(decoration);
        }
    }
    
    function animation(element) {
        const duration = Math.random() * 20 + 10;
        const direction = Math.random() > 0.5 ? 1 : -1;
        
        element.style.animation = `float ${duration}s infinite ease-in-out`;
        
        // Add CSS for float animation
        const floatStyle = document.createElement('style');
        floatStyle.textContent = `
            .bg-decoration {
                position: absolute;
                border-radius: 50%;
                background: linear-gradient(45deg, rgba(74, 111, 165, 0.1), rgba(255, 255, 255, 0.1));
                z-index: -1;
                pointer-events: none;
            }
            
            @keyframes float {
                0% {
                    transform: translateY(0) rotate(0deg);
                }
                50% {
                    transform: translateY(${10 * direction}px) rotate(${10 * direction}deg);
                }
                100% {
                    transform: translateY(0) rotate(0deg);
                }
            }
        `;
        document.head.appendChild(floatStyle);
    }
});