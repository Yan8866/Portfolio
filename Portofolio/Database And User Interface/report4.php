<?php
session_start();
require 'includes/database.php';
$conn = getDB();
$cookieuserID = $_COOKIE['UserID'];

require 'includes/updateauditlog.php';

// get the full name of the current user 
$sql = "SELECT YEAR(S.SaleDate) AS TheYear, 
               COALESCE(SUM(S.Quantity), 0) AS Total_Quantity, 
               (ROUND(COALESCE(SUM(S.Quantity), 0) * 1. / 365)) AS Avg_Units_Sold_Per_Day,
               (
                SELECT COALESCE(SUM(S2.Quantity), 0) 
                FROM sale S2
                WHERE DATE_FORMAT(SaleDate, '%m%d') = '0202'
                AND YEAR(S2.SaleDate) = YEAR(S.SaleDate) ) AS Units_Sold_On_Groundhog_Day
        FROM sale S
        LEFT JOIN product P ON P.PID = S.PID
        LEFT JOIN product_category PC ON P.PID = PC.PID
        WHERE PC.CategoryName = 'Air Conditioning'
        GROUP BY YEAR(S.SaleDate)
        ORDER BY YEAR(S.SaleDate);";

$result = mysqli_query($conn, $sql);

$rows = mysqli_fetch_all($result, MYSQLI_ASSOC);
?>

<?php require 'includes/header.php'; ?>

<table>
  <caption> Report 4 - Air Conditioners on Groundhog Day </caption> <br> <br>
  <thead>
    <tr>
      <th scope="col"> The Year </th>
      <th scope="col"> Total Number of Items Sold</th>
      <th scope="col"> Average Number of Units Sold Per Day </th>
      <th scope="col"> Total Number of Items Sold on Groundhog Day </th>
    </tr>
  </thead>
  <tbody>
  <?php foreach ($rows as $row): ?>
    <tr>
      <th scope="row"> <?php echo $row['TheYear'] ?></th>
      <th scope="row"> <?php echo $row['Total_Quantity'] ?></th>
      <td> <?php echo $row['Avg_Units_Sold_Per_Day'] ?></td>
      <td> <?php echo $row['Units_Sold_On_Groundhog_Day'] ?> </td>
    </tr>
  <?php endforeach; ?> 
</table>

<?php require 'includes/footer.php'; ?>