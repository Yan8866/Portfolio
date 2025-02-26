<?php 
// Capture the report name from the URL
$ReportViewed = isset($_GET['report']) ? $_GET['report'] : '';
//echo $ReportViewed;

// Record the report name as needed (e.g., in the database, session, log file)
if ($ReportViewed) {
    // Store the report name in the session
    $_SESSION['last_viewed_report'] = $ReportViewed;

    // Save the report name to the database with error checking
    $stmt = $conn->prepare("INSERT INTO audit_log (DateAndTime, ReportViewed, EmployeeID) 
                           VALUES (CURRENT_TIMESTAMP(), ?, ?)");
    if ($stmt) {
        $stmt->bind_param("ss", $ReportViewed, $cookieuserID);
        $stmt->execute();
        $stmt->close();  
    }
}
?>