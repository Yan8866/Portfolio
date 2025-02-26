<?php 
error_reporting(E_ERROR | E_PARSE); // Show only errors
ini_set('display_errors', 0); // Do not display errors on the web page
require 'includes/database.php';
require 'includes/url.php';
session_start();
$conn = getDB();
?>

<!DOCTYPE html>
<html>
<head>
    <title>BuzzBuy Data Warehouse</title>
    <meta charset="utf-8">
    <link rel = "stylesheet" href="./login.css" /> 
</head>
<body>

<h1>BuzzBuy Data Warehouse</h1>
<h2>Login</h2>
<form method="post">
        <div class="credential"> EmployeeID: 
            <input name="employeeID" type="text" placeholder = "7 digits" required>  
        </div>
        <div class="credential"> Password: <input name="password" type="password" placeholder = "SSN last four digits-Lastname" required> </div>
        <button> Submit </button>
</form>


<?php
$employeeID = '';
$password = '';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $employeeID = trim($_POST["employeeID"]); 
    $password = trim($_POST["password"]);
    //echo $employeeID. ", ". $password;
}                    // get employeeID and password from the user 

// check the employeeID the user typed is actually stored in the employee table
$sql1 = "SELECT EmployeeID FROM employee; ";
$result1 = mysqli_query($conn, $sql1);
if ($result1 === false) {
echo mysqli_error($conn);
} else {
    $employeeid = []; // Initialize an empty array to hold the employeeID
    while ($row = mysqli_fetch_assoc($result1)) {
        $employeeid[] = $row['EmployeeID'];
        //var_dump($employeeid);
}
}

// get the user password from the database 
$sql2 = "SELECT CONCAT(Last4_SSN, '-', UPPER(LEFT(LName,1)), LOWER(SUBSTRING(LName,2,LENGTH(LName)))) AS password
         FROM employee 
         WHERE EmployeeID = '$employeeID'; "; 

// echo $sql2;

$result2 = mysqli_query($conn, $sql2);

if ($result2 === false) {
echo mysqli_error($conn);
} else {
$passwordg = trim(mysqli_fetch_all($result2)[0][0]);
//echo $passwordg; 
}

//validate the employeeID and password 
if (in_array($employeeID, $employeeid) && $password == $passwordg) {
    $_SESSION['is_logged_in'] = true;
    $_SESSION['employeeID'] = $employeeID;
    setcookie("UserID", $employeeID, time() + 3600, '/');
    // echo $_COOKIE["UserID"];
    header("Location: index.php");;   //redirect to the index page
    exit;
}
?>