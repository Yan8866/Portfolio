<?php

require 'includes/database.php';

$conn = getDB();

// select information about all the holidays in the database  
$sql = "SELECT HDate AS HolidayDate, HName AS HolidayName, AddedByEID 
        FROM holiday
        ORDER BY HDate DESC;";

$result = mysqli_query($conn, $sql);

$rows = mysqli_fetch_all($result, MYSQLI_ASSOC);
?>

<?php require 'includes/header.php'; ?>

<table>
  <caption> Holidays </caption>
  <thead>
    <tr>
      <th scope="col"> HolidayDate </th>
      <th scope="col"> HolidayName </th>
      <th scope="col"> AddedByEID </th>
    </tr>
  </thead>
  <tbody>
  <?php foreach ($rows as $row): ?>
    <tr>
      <th scope="row"> <?php echo $row["HolidayDate"] ?></th>
      <td> <?php echo $row["HolidayName"] ?></td>
      <td> <?php echo $row["AddedByEID"] ?> </td>
    </tr>
  <?php endforeach; ?> 
</table>

<?php require 'includes/footer.php'; ?> 
