//static/js/navigation.js
const socket = io('/host');
let isNavigationInProgress = false;

// Unified navigation handler
function navigateTo(page) {
    if (isNavigationInProgress) return;
    isNavigationInProgress = true;

    // Always notify server of navigation
    socket.emit('navigation_event', { 
        path: page,
        isMobile: /Mobi/i.test(navigator.userAgent)
    });

    // Check if the current page is registration_capture.html
    if (window.location.pathname.includes('registration_capture.html')) {
        // Delay navigation by 1 second
        setTimeout(() => {
            window.location.href = page;
        }, 1000);
    } else {
        // Immediate local navigation
        window.location.href = page;
    }
}

// Server sync handler
socket.on('navigation_sync', (data) => {
    const currentPath = window.location.pathname;
    const targetPath = data.path;

    // Define the ignored transitions
    const ignoredTransitions = [
        ['/dashboard/user', '/dashboard/optimization'],
        ['/dashboard/optimization', '/dashboard/user']
    ];

    // Check if the transition should be ignored
    const isIgnoredTransition = ignoredTransitions.some(([from, to]) => 
        (currentPath.includes(from) && targetPath.includes(to)) || 
        (currentPath.includes(to) && targetPath.includes(from))
    );

    // If the transition is not ignored and the current path doesn't match the target path, sync navigation
    if (!isIgnoredTransition && !currentPath.endsWith(targetPath)) {
        console.log(`Syncing to ${targetPath}`);
        isNavigationInProgress = true;
        window.location.href = targetPath;
    }
});

// Track navigation completion
window.addEventListener('load', () => {
    isNavigationInProgress = false;

    // Notify server of current page
    socket.emit('page_loaded', {
        path: window.location.pathname,
        isMobile: /Mobi/i.test(navigator.userAgent)
    });
});

socket.on('mobile_login', function(data) {
    console.log('mobile login event received:', data);
    const sessionId = data.session_id;  // Get the session ID
    window.location.href = `/dashboard/user?session_id=${encodeURIComponent(sessionId)}`;
});