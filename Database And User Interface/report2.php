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
   <!-- <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous"> -->
    <!--<link rel="stylesheet" href="css/modal.css"> -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>

<body>
    <div id="main_container">
        <div class="center_content">
            <div class="center_left">
                <div class="features">
                    <div class="profile_section p-5">
                        <h2 style="color: #24527a; font-size: 2em; text-align: center;">Report 2: Category Report</h2>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th class="heading">Category Name</th>
                                    <th class="heading">Total Number of Products</th>
                                    <th class="heading">Total Number of Manufacturers</th>
                                    <th class="heading">Average Retail Price</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php
                                $sql = "SELECT CategoryName, COUNT(DISTINCT PID) AS NumOfProd, \n"
                                    . "COUNT(DISTINCT Manufacturer) AS NumofManu, ROUND(AVG(RetailPrice),2) AS AvgPrice \n"
                                    . "FROM product NATURAL JOIN product_category\n"
                                    . "GROUP BY CategoryName\n"
                                    . "ORDER BY CategoryName;";

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
                                    echo "<td>{$row['CategoryName']}</td>";
                                    echo "<td>{$row['NumOfProd']}</td>";
                                    echo "<td>{$row['NumofManu']}</td>";
                                    echo "<td>{$row['AvgPrice']}</td>";
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