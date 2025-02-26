<?php
session_start();

require 'includes/database.php';
require 'includes/header.php';
$conn = getDB();
$cookieuserID = $_COOKIE['UserID'];
require 'includes/auth.php';

if (!isLoggedIn()) {
    die("Unauthorized");
}
require 'includes/updateauditlog.php'; ?>

<head>
    <meta charset="UTF-8">
    <title>Special Privileges Page</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css"> 
</head>

<body>
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-8 offset-md-2">
                <h2 class="text-center mb-4" style="color: #24527a; font-size: 2em; text-align: center;">Holiday List</h2>
                <table class="table">
                    <!-- Table header -->
                    <thead>
                        <tr>
                            <th>Holiday Date</th>
                            <th>Holiday Name</th>
                        </tr>
                    </thead>
                    <!-- Table body -->
                    <tbody>
                        <!-- PHP code to display holidays -->
                        <?php
                        // Handle POST request to add holiday
                        if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['holidayDate']) && isset($_POST['holidayName'])) {
                            $holidayDate = mysqli_real_escape_string($conn, $_POST['holidayDate']);
                            $holidayName = mysqli_real_escape_string($conn, $_POST['holidayName']);
                            $addedByEID = mysqli_real_escape_string($conn, $_SESSION['employeeID']); // Assuming employee ID is stored in session

                            // Check if holiday already exists
                            $check_query = "SELECT * FROM holiday WHERE HDate = '$holidayDate' AND HName = '$holidayName'";
                            $check_result = mysqli_query($conn, $check_query);

                            if (mysqli_num_rows($check_result) > 0) {
                                // Holiday already exists
                                echo "<tr><td colspan='2'>Holiday already exists!</td></tr>";
                            } else {
                                // Insert new holiday into database
                                $insert_query = "INSERT INTO holiday (HDate, HName, AddedByEID) VALUES ('$holidayDate', '$holidayName', '$addedByEID')";

                                if (mysqli_query($conn, $insert_query)) {
                                    // Success message and redirect
                                    echo "<script>alert('Holiday added successfully!'); window.location.href = '" . $_SERVER['PHP_SELF'] . "';</script>";
                                } else {
                                    // Handle specific MySQL error
                                    if (mysqli_errno($conn) == 1452) {
                                        echo "<tr><td colspan='2'>Error: Cannot add duplicate holiday. Please try again with a different date or name.</td></tr>";
                                    } else {
                                        echo "<tr><td colspan='2'>Error: " . mysqli_error($conn) . "</td></tr>";
                                    }
                                }
                            }
                        }

                        // Fetch all_districts_assigned value
                        $cookieemployeeID = mysqli_real_escape_string($conn, $_SESSION['employeeID']);
                        $query = "SELECT COUNT(DISTINCT D.DistrictNumber) = COUNT(A.DistrictNumber) AS all_districts_assigned
                                  FROM District D
                                  LEFT JOIN (
                                      SELECT DISTINCT DistrictNumber
                                      FROM Assigned A
                                      INNER JOIN Employee E ON A.EmployeeID = E.EmployeeID
                                      WHERE E.EmployeeID = '$cookieemployeeID'
                                  ) AS A ON D.DistrictNumber = A.DistrictNumber";

                        $result = mysqli_query($conn, $query);

                        if ($result) {
                            $row = mysqli_fetch_assoc($result);
                            $all_districts_assigned = $row['all_districts_assigned']; // Assign the value
                        } else {
                            $error_msg = "Query failed: " . mysqli_error($conn);
                        }

                        // Fetch holiday data
                        $holiday_query = "SELECT HDate AS holiday_date, HName AS holiday_name FROM holiday";
                        $holiday_result = mysqli_query($conn, $holiday_query);

                        if (!$holiday_result) {
                            $error_msg = "Holiday query failed: " . mysqli_error($conn);
                        }

                        // Display holiday rows
                        if ($holiday_result && mysqli_num_rows($holiday_result) > 0) {
                            while ($holiday_row = mysqli_fetch_assoc($holiday_result)) {
                                echo "<tr>";
                                echo "<td>" . htmlspecialchars($holiday_row['holiday_date']) . "</td>";
                                echo "<td>" . htmlspecialchars($holiday_row['holiday_name']) . "</td>";
                                echo "</tr>";
                            }
                        } else {
                            echo "<tr><td colspan='2'>No holidays found.</td></tr>";
                        }

                        mysqli_close($conn);
                        ?>
                    </tbody>
                </table>

                <!-- Add Holiday Button -->
                <?php if ($all_districts_assigned == 1) : ?>
                    <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#addHolidayModal">
                        Add Holiday
                    </button>
                <?php endif; ?>
            </div>
        </div>
    </div>

    <!-- Add Holiday Modal -->
    <div class="modal fade" id="addHolidayModal" tabindex="-1" role="dialog" aria-labelledby="addHolidayModalLabel" aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <form id="addHolidayForm" method="post" action="<?php echo htmlspecialchars($_SERVER['PHP_SELF']); ?>">
                    <div class="modal-header" style="background-color: #f1f1f1;">
                        <h5 class="modal-title" id="addHolidayModalLabel">Add Holiday</h5>
                        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body" style="padding: 20px;">
                        <div class="form-group">
                            <label for="holidayDate" style="font-weight: bold;">Holiday Date</label>
                            <input type="date" class="form-control" id="holidayDate" name="holidayDate" required style="width: 100%;">
                        </div>
                        <div class="form-group">
                            <label for="holidayName" style="font-weight: bold;">Holiday Name</label>
                            <input type="text" class="form-control" id="holidayName" name="holidayName" required style="width: 100%;">
                        </div>
                    </div>
                    <div class="modal-footer" style="background-color: #f1f1f1; border-top: 1px solid #e9ecef;">
                        <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                        <button type="submit" class="btn btn-primary">Add Holiday</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- JavaScript to handle modal and form submission -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        $(document).ready(function() {
            // Hide modal when closed
            $('#addHolidayModal').on('hidden.bs.modal', function() {
                $(this).find('form').trigger('reset');
            });

            // Show modal on page load if form was submitted
            <?php if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['holidayDate']) && isset($_POST['holidayName'])) : ?>
                $('#addHolidayModal').modal('show');
            <?php endif; ?>
        });
    </script>
</body>

</html>