<?php
require 'includes/database.php';
require 'includes/auth.php';
require 'includes/header.php';

session_start();

if (!isLoggedIn()) {
    die("Unauthorized");
}

$conn = getDB();
$cookieuserID = $_COOKIE['UserID'];

$date_sql = "SELECT DISTINCT DATE_FORMAT(SaleDate, '%Y-%m') AS YearMonth FROM sale ORDER BY YEAR(SaleDate);";
$date_result = mysqli_query($conn, $date_sql);
if (!empty($date_result) && (mysqli_num_rows($date_result) == 0)) {
    array_push($error_msg,  "SELECT ERROR: <br>" . __FILE__ . " line:" . __LINE__);
}
if ($date_result === false) {
    echo mysqli_error($conn);
    return;
}

$selected_date = '';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $selected_date = $_POST['selected_date'];
}
?>


<html>

<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" href="css/modal.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>

<body>
    <div id="main_container">
        <div class="center_content">
            <div class="center_left">
                <div class="features">
                    <div class="profile_section container mt-5">
                        <h2 style="color: #24527a; font-size: 2em; text-align: center;">
                            Report 6 - District with Highest Volume for each Category
                        </h2>

                        <div class="container p-3">
                            <form method="post">
                                <div class="form-group">
                                    <label for="dropdown" style="font-size: 20px">Select a Year and Month:</label>
                                    <select class="form-control" id="dropdown" name="selected_date" onchange="this.form.submit()">
                                        <option style="font-size: 20px"></option>
                                        <?php
                                        while ($row = mysqli_fetch_array($date_result, MYSQLI_ASSOC)) {
                                            $selected = ($row['YearMonth'] == $selected_date) ? 'selected' : '';
                                            print "<option value=" . $row['YearMonth'] . " $selected>" . $row['YearMonth'] . "</option>";
                                        }
                                        ?>
                                    </select>
                                </div>
                            </form>

                            <?php if ($selected_date) : ?>
                                <table class="table">
                                    <thead>
                                        <tr>
                                            <th class="heading">Category Name</th>
                                            <th class="heading">District Number</th>
                                            <th class="heading">Units Sold</th>
                                            <th class="heading">Drilldown</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php
                                        $selected_date_sql = "SELECT Category, DistrictNumber, UnitsSold
                                                                FROM
                                                                    (SELECT unitsCategory AS Category, MAX(units) AS UnitsSold
                                                                    FROM
                                                                        (SELECT CAT.CategoryName AS unitsCategory, ST.DistrictNumber, SUM(Quantity) AS units
                                                                        FROM sale SA
                                                                        JOIN product P
                                                                        ON SA.PID = P.PID
                                                                        JOIN store ST
                                                                        ON SA.StoreNumber = ST.StoreNumber
                                                                        JOIN product_category PC
                                                                        ON SA.PID = PC.PID
                                                                        JOIN category CAT
                                                                        ON CAT.CategoryName = PC.CategoryName
                                                                        WHERE DATE_FORMAT(SaleDate, '%Y-%m') = '" . $selected_date . "'
                                                                        GROUP BY CAT.CategoryName, ST.DistrictNumber) AS Q1
                                                                    GROUP BY unitsCategory) AS Q2
                                                                    JOIN
                                                                        (SELECT CAT.CategoryName AS DisCategory, ST.DistrictNumber AS DistrictNumber, SUM(Quantity) AS DisUnits
                                                                        FROM sale SA
                                                                        JOIN product P
                                                                        ON SA.PID = P.PID
                                                                        JOIN store ST
                                                                        ON SA.StoreNumber = ST.StoreNumber
                                                                        JOIN product_category PC
                                                                        ON SA.PID = PC.PID
                                                                        JOIN category CAT
                                                                        ON CAT.CategoryName = PC.CategoryName
                                                                        WHERE DATE_FORMAT(SaleDate, '%Y-%m') = '" . $selected_date . "'
                                                                        GROUP BY CAT.CategoryName, ST.DistrictNumber) AS Q3
                                                                        ON Category = DisCategory AND UnitsSold = DisUnits
                                                                ORDER BY Category;";


                                        $selected_date_result = mysqli_query($conn, $selected_date_sql);
                                        if (!empty($selected_date_result) && (mysqli_num_rows($selected_date_result) == 0)) {
                                            array_push($error_msg,  "SELECT ERROR: <br>" . __FILE__ . " line:" . __LINE__);
                                        }
                                        if ($selected_date_result === false) {
                                            echo mysqli_error($conn);
                                            return;
                                        }
                                        while ($row = mysqli_fetch_array($selected_date_result, MYSQLI_ASSOC)) {
                                            echo "<tr>";
                                            echo "<td>{$row['Category']}</td>";
                                            echo "<td>{$row['DistrictNumber']}</td>";
                                            echo "<td>{$row['UnitsSold']}</td>";

                                            // Drilldown 
                                            $drilldown_sql = "SELECT ST.StoreNumber, ST.StateName, ST.CityName
                                                FROM store ST
                                                JOIN city C ON ST.CityName = C.CityName AND ST.StateName = C.StateName
                                                JOIN sale SA ON SA.StoreNumber = ST.StoreNumber
                                                JOIN product P ON SA.PID = P.PID
                                                JOIN product_category PC ON P.PID = PC.PID
                                                JOIN category CAT ON PC.CategoryName = CAT.CategoryName
                                                WHERE ST.DistrictNumber = " . $row['DistrictNumber'] . "
                                                AND DATE_FORMAT(SA.SaleDate, '%Y-%m') = '" . $selected_date . "'
                                                AND CAT.CategoryName = '" . $row['Category'] . "'
                                                ORDER BY CAST(ST.StoreNumber AS INT) ASC;";


                                            $drilldown_result = mysqli_query($conn, $drilldown_sql);
                                            if ($drilldown_result === false) {
                                                echo mysqli_error($conn);
                                                return;
                                            }
                                            echo '<td>
                                                    <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#exampleModalCenter-' . preg_replace('/\s.+/', '', $row['DistrictNumber']) . '">
                                                        View Drilldown
                                                    </button>
                                                    <!-- Modal -->
                                                    <div class="modal bd-example-modal-lg" id="exampleModalCenter-' . preg_replace('/\s.+/', '', $row['DistrictNumber']) . '" tabindex="-1" role="dialog" aria-labelledby="exampleModalCenterTitle" aria-hidden="true">
                                                        <div class="modal-dialog" role="document">
                                                            <div class="modal-content">
                                                                <div class="modal-header">
                                                                    <h5 class="modal-title" id="exampleModalLongTitle">' . $row['Category'] . ' Sales in District #' . $row['DistrictNumber'] . ' on ' . $selected_date . '</h5>
                                                                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                                                    <span aria-hidden="true">&times;</span>
                                                                    </button>
                                                                </div>
                                                                <div class="modal-body">
                                                                    <table class="table table-bordered">
                                                                        <thead>
                                                                            <tr>
                                                                                <th class="heading">Store Number</th>
                                                                                <th class="heading">State Name</th>
                                                                                <th class="heading">City Name</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody>';
                                            while ($drilldown_row = mysqli_fetch_array($drilldown_result, MYSQLI_ASSOC)) {
                                                echo "<tr>";
                                                echo "<td>{$drilldown_row['StoreNumber']}</td>";
                                                echo "<td>{$drilldown_row['StateName']}</td>";
                                                echo "<td>{$drilldown_row['CityName']}</td>";
                                                echo "</tr>";
                                            }
                                            echo '</tbody>
                                                                    </table>
                                                                    
                                                                </div>
                                                                <div class="modal-footer">
                                                                    <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </td>';
                                            echo "</tr>";
                                        }
                                        ?>
                                    </tbody>
                                </table>

                                <?php
                                // Update audit log here, after the report has been generated
                                require 'includes/updateauditlog.php';
                                ?>

                            <?php endif; ?>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>

</html>

<?php require 'includes/footer.php'; ?>