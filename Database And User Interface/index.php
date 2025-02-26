<?php 
session_start();
require 'includes/database.php';
require 'includes/auth.php';
require 'includes/updateauditlog.php';
$conn = getDB();

if (!isLoggedIn()) {
        header("Location: login.php");
}

$cookieuserID = $_COOKIE['UserID']; 

// get the full name of the current user 
$sql = "SELECT CONCAT(FName, ' ', UPPER(LEFT(LName,1)), LOWER(SUBSTRING(LName,2,LENGTH(LName)))) AS FullName 
        FROM employee 
        WHERE EmployeeID = '$cookieuserID'; ";

$result = mysqli_query($conn, $sql);
if ($result === false) {
echo mysqli_error($conn);
} else {
$fullname = mysqli_fetch_all($result)[0][0];
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>BuzzBuy Data Warehouse</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="./index.css" /> 
</head>
<body>
    <header>
        <h1>BuzzBuy Data Warehouse</h1>
    </header>

<?php if (isLoggedIn()): ?>
    <div class="message"> <?php  echo "Welcome, $fullname!"; ?> </div> 
    <p class="login">You are logged in. <a class = "log" href="logout.php" style="font-size:28px;">Log out</a></p>
     <!-- add content only available for logged in users -->
<?php else: ?>
    <p class="login">You are not logged in. <a class = "log" href="login.php">Log in</a></p>
<?php endif; ?>

<?php
$sql2 = "SELECT COUNT(*) AS store_count  FROM store; ";
$result2 = mysqli_query($conn, $sql2);
if ($result2 === false) {
echo mysqli_error($conn);
} else {
$storecount = mysqli_fetch_all($result2)[0][0];
}

$sql3 = "SELECT COUNT(*) AS city_count  FROM city; ";
$result3 = mysqli_query($conn, $sql3);
if ($result3 === false) {
echo mysqli_error($conn);
} else {
$citycount = mysqli_fetch_all($result3)[0][0];
}

$sql4 = "SELECT COUNT(*) AS district_count FROM district; ";
$result4 = mysqli_query($conn, $sql4);
if ($result4 === false) {
echo mysqli_error($conn);
} else {
$districtcount = mysqli_fetch_all($result4)[0][0];
}

$sql5 = "SELECT COUNT( DISTINCT ManufacturerName) AS manu_count FROM manufacturer; ";
$result5 = mysqli_query($conn, $sql5);
if ($result5 === false) {
echo mysqli_error($conn);
} else {
$manu_count = mysqli_fetch_all($result5)[0][0];
}

$sql6 = "SELECT COUNT(*) AS product_Count FROM product; ";
$result6 = mysqli_query($conn, $sql6);
if ($result6 === false) {
echo mysqli_error($conn);
} else {
$productcount = mysqli_fetch_all($result6)[0][0];
}

$sql7 = "SELECT COUNT( DISTINCT CategoryName) AS category_count FROM product_category; ";
$result7 = mysqli_query($conn, $sql7);
if ($result7 === false) {
echo mysqli_error($conn);
} else {
$categorycount = mysqli_fetch_all($result7)[0][0];
}

$sql8 = "SELECT COUNT(*) AS holiday_count FROM holiday; ";
$result8 = mysqli_query($conn, $sql8);
if ($result8 === false) {
echo mysqli_error($conn);
} else {
$holidaycount = mysqli_fetch_all($result8)[0][0];
}
?>

<fieldset>
        <legend>Statistics</legend>
        <div class="stat"> Store Count:        <?php echo  $storecount ?> </div>
        <div class="stat"> City Count:         <?php echo  $citycount ?> </div>
        <div class="stat"> District Count:     <?php echo  $districtcount ?> </div>
        <div class="stat"> Manufacturer Count: <?php echo  $manu_count ?> </div>
        <div class="stat"> Product Count:      <?php echo  $productcount ?> </div>
        <div class="stat"> Category Count:     <?php echo  $categorycount ?> </div>
        <div class="stat"> Holiday Count:      <?php echo  $holidaycount ?> </div>
</fieldset>

<!--find the total number of districts and the number of districts that is assigned to the current user   -->
<?php
$sql9 = "SELECT COUNT(DISTINCT DistrictNumber) AS Num_of_AssignedDistricts
         FROM assigned WHERE EmployeeID = '$cookieuserID';";
$result9 = mysqli_query($conn, $sql9);
if ($result9 === false) {
        echo mysqli_error($conn);
        } else {
        $num_of_AssignnedDistricts = mysqli_fetch_all($result9)[0][0];
        // echo  $num_of_AssignnedDistricts; 
        }

$sql10 = "SELECT COUNT(DISTINCT DistrictNumber) AS Num_of_AllDistricts FROM assigned; ";
$result10 = mysqli_query($conn, $sql10);
if ($result10 === false) {
       echo mysqli_error($conn);
       } else {
       $num_of_AllDistricts = mysqli_fetch_all($result10)[0][0];
       // echo $num_of_AllDistricts; 
       }
?>   

<!--Display reports and Add Holiday button accordingly -->
<?php if ($num_of_AssignnedDistricts == $num_of_AllDistricts): ?>
<fieldset>
        <legend>Reports</legend>
        <div> <a href="report1.php?report=<?php echo urlencode("Manufacturer's Product Report"); ?>">
        Report 1 - Manufacturer's Product Report </a> 
        </div>
        <div> <a href="report2.php?report=<?php echo urlencode('Category Report'); ?>"> 
                Report 2 - Category Report </a> </div>
        <div> <a href="report3.php?report=<?php echo urlencode('Actual versus Predicted Revenue for GPS units'); ?>"> 
        Report 3 - Actual vs Predicted Revenue-GPS Units </a> 
        </div>
        <div> <a href="report4.php?report=<?php echo urlencode('Air Conditioners on Groundhog Day?'); ?>"> 
        Report 4 - Air Conditioners on Groundhog Day? </a> 
        </div>
        <div> <a href="report5.php?report=<?php echo urlencode('Store Revenue by Year by State'); ?>"> 
        Report 5 - Store Revenue by Year by State </a> 
        </div>
        <div> <a href="report6.php?report=<?php echo urlencode('District with Highest Volume for each Category'); ?>">
        Report 6 - District with Highest Volume for each Category </a> 
        </div>
        <div> <a href="report7.php?report=<?php echo urlencode('Revenue by Population'); ?>"> 
        Report 7 - Revenue by Population </a> </div>
</fieldset>
<div> <a href="addholiday.php"> Add Holiday </a> </div>
<?php else: ?>
<fieldset>
        <legend>Reports</legend>
        <div> <a href="report1.php?report=<?php echo urlencode("Manufacturer's Product Report"); ?>"> 
                Report 1 - Manufacturer's Product Report </a> 
        </div>
        <div> <a href="report2.php?report=<?php echo urlencode('Category Report'); ?>"> Report 2 - Category Report </a> </div>
        <div> <a href="report3.php?report=<?php echo urlencode('Actual versus Predicted Revenue for GPS units'); ?>"> 
                Report 3 - Actual vs Predicted Revenue-GPS units </a> 
        </div>
        <div> <a href="report4.php?report=<?php echo urlencode('Air Conditioners on Groundhog Day?'); ?>"> 
                Report 4 - Air Conditioners on Groundhog Day? </a> 
        </div>
</fieldset>
<?php endif; ?>

<div> <a href="viewholiday.php"> View Holiday </a> </div>

<!-- check if the current user can view audit log -->
<?php
$sql11 = "SELECT AuditLogFlag  FROM employee WHERE EmployeeID = $cookieuserID; ";
$result11 = mysqli_query($conn, $sql11);
if ($result11 === false) {
        echo mysqli_error($conn);
        } else {
        $auditlogflag = mysqli_fetch_all($result11)[0][0];
        //echo  $auditlogflag; 
        }
?>   

<!-- show the View Audit Log link accordingly -->
<?php if ($auditlogflag == 1): ?>
<div> <a href="auditlog.php"> View Audit Log </a> </div> <br> <Br>
<?php endif; ?>

<?php require 'includes/footer.php'; ?>