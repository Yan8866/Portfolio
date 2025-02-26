<?php
require 'includes/database.php';
require 'includes/auth.php';
require 'includes/header.php';
session_start();

$conn = getDB();
$cookieuserID = $_COOKIE['UserID'];

require 'includes/updateauditlog.php';
?>

<html>
<head>
    <!-- <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous"> -->
    <!-- <link rel="stylesheet" href="css/modal.css"> -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>

<body>
    <div id = "main_container">
        <div class = "center_content">
            <div class = "center_left">
                <div class = "features">
                    <div class = "profile_section p-5">
                        <h2 style="color: #24527a; font-size: 2em; text-align: center;">
                          Report 7 - Revenue by Population
                        </h2>
                        <table class = "table">
                            <thead>
                                <tr>
                                    <th class = "heading">Year</th>
                                    <th class = "heading">City Category</th>
                                    <th class = "heading">Average Revenue</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php
                                $sql = "SELECT YEAR(SaleDate) AS TheYear,
                                        CASE
                                            WHEN PopulationSize < 3700000 THEN 'Small'
                                            WHEN PopulationSize >= 3700000 AND PopulationSize < 6700000 THEN 'Medium'
                                            WHEN PopulationSize >= 6700000 AND PopulationSize < 9000000 THEN 'Large'
                                            WHEN PopulationSize >= 9000000 THEN 'ExtraLarge'
                                            END AS CitySize, ROUND(AVG(Quantity * IFNULL(DiscountPrice, RetailPrice)), 2) AS AvgRevenue
                                        FROM sale SA
                                        JOIN product P
                                        ON SA.PID = P.PID
                                        JOIN store ST
                                        ON SA.StoreNumber = ST.StoreNumber
                                        JOIN city C
                                        ON ST.CityName = C.CityName AND ST.StateName = C.StateName
                                        LEFT JOIN discount D
                                        ON SaleDate = DiscountedDate AND SA.PID = D.PID
                                        GROUP BY TheYear, CitySize
                                        ORDER BY TheYear, PopulationSize;";

                                $result = mysqli_query($conn, $sql);
                                if (!empty($result) && (mysqli_num_rows($result) == 0)) {
                                    array_push($error_msg, "SELECT ERROR: <br>" . __FILE__ . "line:" . __FILE__);                      
                                }
                                if ($result == false) {
                                    echo mysqli_error($conn);
                                    return;
                                }
                                while ($row = mysqli_fetch_array($result, MYSQLI_ASSOC)) {
                                    echo "<tr>";
                                    echo "<td>{$row['TheYear']}</td>";
                                    echo "<td>{$row['CitySize']}</td>";
                                    echo "<td>{$row['AvgRevenue']}</td>";
                                }
                                ?>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>

