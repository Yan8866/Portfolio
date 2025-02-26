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

require 'includes/updateauditlog.php';
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
                    <div class="profile_section p-5">
                    <h2 style="color: #24527a; font-size: 2em; text-align: center;"> Report 1-Manufacturer's Product Report </h2>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th class="heading">Manufacturer Name</th>
                                    <th class="heading">Number of Products</th>
                                    <th class="heading">Average Price</th>
                                    <th class="heading">Min Price</th>
                                    <th class="heading">Max Price</th>
                                    <th class="heading">Drilldown</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php
                                $sql = "SELECT Manufacturer, COUNT(DISTINCT PID) AS NumOfProducts, \n"
                                    . "ROUND(AVG(RetailPrice),2) AS AvgPrice, \n"
                                    . "MIN(RetailPrice) AS MinPrice, \n"
                                    . "MAX(RetailPrice) AS MaxPrice\n"
                                    . "FROM product\n"
                                    . "GROUP BY Manufacturer\n"
                                    . "ORDER BY  AvgPrice  DESC\n"
                                    . "LIMIT 100;";

                                $result = mysqli_query($conn, $sql);
                                if (!empty($result) && (mysqli_num_rows($result) == 0)) {
                                    array_push($error_msg,  "SELECT ERROR: <br>" . __FILE__ . " line:" . __LINE__);
                                }
                                if ($result === false) {
                                    echo mysqli_error($conn);
                                    return;
                                }
                                while ($row = mysqli_fetch_array($result, MYSQLI_ASSOC)) {
                                    echo "<tr>";
                                    echo "<td>{$row['Manufacturer']}</td>";
                                    echo "<td>{$row['NumOfProducts']}</td>";
                                    echo "<td>{$row['AvgPrice']}</td>";
                                    echo "<td>{$row['MinPrice']}</td>";
                                    echo "<td>{$row['MaxPrice']}</td>";

                                    // Drilldown 
                                    $drilldown_sql = "SELECT PR.PID,PName, RetailPrice,\n"
                                    . "GROUP_CONCAT(CategoryName SEPARATOR \", \") AS Category\n"
                                    . "FROM product AS PR \n"
                                    . "JOIN product_category AS PC ON PR.PID = PC.PID\n"
                                    . "WHERE Manufacturer = \"".$row['Manufacturer']."\"\n"
                                    . "GROUP BY PR.PID\n"
                                    . "ORDER BY RetailPrice DESC;";
                                    $drilldown_result = mysqli_query($conn, $drilldown_sql);
                                    if (!empty($drilldown_result) && (mysqli_num_rows($drilldown_result) == 0)) {
                                        array_push($error_msg,  "SELECT ERROR: <br>" . __FILE__ . " line:" . __LINE__);
                                    }
                                    if ($drilldown_result === false) {
                                        echo mysqli_error($conn);
                                        return;
                                    }
                                    echo '<td>
                                        <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#exampleModalCenter-'. preg_replace('/\s.+/', '', $row['Manufacturer']) .'">
                                            View Drilldown
                                        </button>
                                        <!-- Modal -->
                                        <div class="modal bd-example-modal-lg" id="exampleModalCenter-'. preg_replace('/\s.+/', '', $row['Manufacturer']) .'" tabindex="-1" role="dialog" aria-labelledby="exampleModalCenterTitle" aria-hidden="true">
                                            <div class="modal-dialog" role="document">
                                                <div class="modal-content">
                                                    <div class="modal-header">
                                                        <h5 class="modal-title" id="exampleModalLongTitle">'. $row['Manufacturer'] .' Products</h5>
                                                        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                                        <span aria-hidden="true">&times;</span>
                                                        </button>
                                                    </div>
                                                    <div class="modal-body">
                                                        <table class="table table-bordered">
                                                            <thead>
                                                                <tr>
                                                                    <th class="heading">PID</th>
                                                                    <th class="heading">Name</th>
                                                                    <th class="heading">Categories</th>
                                                                    <th class="heading">Price</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>';
                                                                while($drilldown_row = mysqli_fetch_array($drilldown_result, MYSQLI_ASSOC)) {
                                                                    echo "<tr>";
                                                                    echo "<td>{$drilldown_row['PID']}</td>";
                                                                    echo "<td>{$drilldown_row['PName']}</td>";
                                                                    echo "<td>{$drilldown_row['Category']}</td>";
                                                                    echo "<td>{$drilldown_row['RetailPrice']}</td>";
                                                                    echo "</tr>";
                                                                }
                                                            echo'</tbody>
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
                    </div>
                </div>
            </div>
            <div class="clear"></div>
        </div>
    </div>
</body>
</html>


<?php require 'includes/footer.php'; ?>