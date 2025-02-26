<?php
session_start();

require 'includes/database.php';
require 'includes/header.php';
$conn = getDB();
$cookieuserID = $_COOKIE['UserID'];
require 'includes/updateauditlog.php'; ?>


<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Special Privileges Page</title>
    <!-- <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css"> -->
    <style>
        .highlight {
            background-color: yellow;
            /* Highlight color */
        }

        .modal-body input {
            width: 100%;
        }
    </style>
</head>

<body>
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-8 offset-md-2">
                <h2 class="text-center mb-4" style="color: #24527a; font-size:2em; text-align: center;">
                    Audit Log
                </h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Date and Time</th>
                            <th>Employee ID</th>
                            <th>Full Name</th>
                            <th>Report Viewed</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php
                        // Fetch employee IDs assigned to all districts
                        $highlight_ids_query = "SELECT E.EmployeeID
                                                FROM employee E
                                                WHERE (SELECT COUNT(DISTINCT D.DistrictNumber) = COUNT(A.DistrictNumber)
                                                       FROM District D
                                                       LEFT JOIN Assigned A ON D.DistrictNumber = A.DistrictNumber AND A.EmployeeID = E.EmployeeID) = 1";
                        $highlight_ids_result = mysqli_query($conn, $highlight_ids_query);
                        $highlight_ids = array();

                        if ($highlight_ids_result && mysqli_num_rows($highlight_ids_result) > 0) {
                            while ($row = mysqli_fetch_assoc($highlight_ids_result)) {
                                $highlight_ids[] = $row['EmployeeID'];
                            }
                        }

                        // Fetch audit log data
                        $audit_log_query = "SELECT DateAndTime, EmployeeID, CONCAT(LName, ', ', FName) AS FullName, ReportViewed
                                           FROM employee NATURAL JOIN audit_log
                                           ORDER BY DateAndTime DESC
                                           LIMIT 100";
                        $audit_log_result = mysqli_query($conn, $audit_log_query);

                        if (!$audit_log_result) {
                            $error_msg = "Audit log query failed: " . mysqli_error($conn);
                        }

                        // Display audit log rows
                        if ($audit_log_result && mysqli_num_rows($audit_log_result) > 0) {
                            while ($audit_log_row = mysqli_fetch_assoc($audit_log_result)) {
                                $highlight_class = in_array($audit_log_row['EmployeeID'], $highlight_ids) ? 'highlight' : '';
                                echo "<tr class='$highlight_class'>";
                                echo "<td>" . htmlspecialchars($audit_log_row['DateAndTime']) . "</td>";
                                echo "<td>" . htmlspecialchars($audit_log_row['EmployeeID']) . "</td>";
                                echo "<td>" . htmlspecialchars($audit_log_row['FullName']) . "</td>";
                                echo "<td>" . htmlspecialchars($audit_log_row['ReportViewed']) . "</td>";
                                echo "</tr>";
                            }
                        } else {
                            echo "<tr><td colspan='4'>No audit logs found.</td></tr>";
                        }

                        mysqli_close($conn);
                        ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- JavaScript -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.bundle.min.js"></script>
</body>

</html>