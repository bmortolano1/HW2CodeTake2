%% Question 1

Question1Data = csvread("./Part1DataTake2.csv");

Q1.i = Question1Data(:,1)
Q1.test_e_weighted = Question1Data(:,2)
Q1.train_e_weighted = Question1Data(:,3)
Q1.test_e_unweighted = Question1Data(:,4)
Q1.train_e_unweighted = Question1Data(:,5)
Q1.test_e_global = Question1Data(:,6)
Q1.train_e_global = Question1Data(:,7)

f1 = figure(1)
plot(Q1.test_e_weighted)
title("Weighted Test Error")
ylabel("Error")
xlabel("Iteration")
saveas(f1, "Q1WeightTestErr.jpg")

f2 = figure(2)
plot(Q1.train_e_weighted)
title("Weighted Train Error")
ylabel("Error")
xlabel("Iteration")
saveas(f2, "Q1WeightTrainErr.jpg")

f3 = figure(3)
plot(Q1.test_e_unweighted)
title("Unweighted Test Error")
ylabel("Error")
xlabel("Iteration")
saveas(f3, "Q1UnweightTestErr.jpg")

f4 = figure(4)
plot(Q1.train_e_unweighted)
title("Unweighted Train Error")
ylabel("Error")
xlabel("Iteration")
saveas(f4, "Q1UnweightTrainErr.jpg")

f5 = figure(5)
plot(Q1.test_e_global)
title("Combined/Boosted Test Error")
ylabel("Error")
xlabel("Iteration")
saveas(f5, "Q1GlobalTestErr.jpg")

f6 = figure(6)
plot(Q1.train_e_global)
title("Combined/Boosted Train Error")
ylabel("Error")
xlabel("Iteration")
saveas(f6, "Q1GlobalTrainErr.jpg")

%% Question 2

Question2Data = csvread("./Part2Data.csv")
Q2.i = Question2Data(:,1)
Q2.test_e_global = Question2Data(:,2)
Q2.train_e_global = Question2Data(:,3)

f7 = figure(7)
plot(Q2.test_e_global)
title("Combined/Boosted Test Error")
ylabel("Error")
xlabel("Iteration")
saveas(f7, "Q2GlobalTestErr.jpg")

f8 = figure(8)
plot(Q2.train_e_global)
title("Combined/Boosted Train Error")
ylabel("Error")
xlabel("Iteration")
saveas(f8, "Q2GlobalTrainErr.jpg")

%% Question 3

Question3Data = csvread("./Part3Data.csv")

Q3.i = Question3Data(:,1)
Q3.single_bias = Question3Data(:,2);
Q3.single_var = Question3Data(:,3);
Q3.agg_bias = Question3Data(:,4);
Q3.agg_var = Question3Data(:,5);

f12 = figure(12)
plot(Q3.single_bias)
hold on
yline(mean(Q3.single_bias))
title("Single Tree Bias")
ylabel("Bias")
xlabel("Iteration")
saveas(f12, "Q3SingleBias.jpg")

f9 = figure(9)
plot(Q3.single_var)
hold on
yline(mean(Q3.single_var))
title("Single Tree Variance")
ylabel("Variance")
xlabel("Iteration")
saveas(f9, "Q3SingleVariance.jpg")

f10 = figure(10)
plot(Q3.agg_bias)
hold on
yline(mean(Q3.agg_bias))
title("Aggregate Predictor Bias")
ylabel("Bias")
xlabel("Iteration")
saveas(f10, "Q3AggBias.jpg")

f11 = figure(11)
plot(Q3.agg_var)
hold on
yline(mean(Q3.agg_var))
title("Aggregate Predictor Variance")
ylabel("Variance")
xlabel("Iteration:")
saveas(f11, "Q3AggVariance.jpg")

%% Question 4 w/ Two Attributes

Question4aData = csvread("./Part4Data2-2.csv")
Q4a.i = Question4aData(:,1)
Q4a.test_e_global = Question4aData(:,2)
Q4a.train_e_global = Question4aData(:,3)

f13 = figure(13)
plot(Q4a.test_e_global)
title("Combined/Boosted Test Error - 2 Attributes")
ylabel("Error")
xlabel("Iteration")
saveas(f13, "Q4aGlobalTestErr.jpg")

f14 = figure(14)
plot(Q4a.train_e_global)
title("Combined/Boosted Train Error - 2 Attributes")
ylabel("Error")
xlabel("Iteration")
saveas(f14, "Q4aGlobalTrainErr.jpg")

%% Question 4 w/ Four Attributes

Question4bData = csvread("./Part4Data4-2.csv")
Q4b.i = Question4bData(:,1)
Q4b.test_e_global = Question4bData(:,2)
Q4b.train_e_global = Question4bData(:,3)

f15 = figure(15)
plot(Q4b.test_e_global)
title("Combined/Boosted Test Error - 4 Attributes")
ylabel("Error")
xlabel("Iteration")
saveas(f15, "Q4bGlobalTestErr.jpg")

f16 = figure(16)
plot(Q4b.train_e_global)
title("Combined/Boosted Train Error - 4 Attributes")
ylabel("Error")
xlabel("Iteration")
saveas(f16, "Q4bGlobalTrainErr.jpg")

%% Question 4 w/ Four Attributes

Question4cData = csvread("./Part4Data6-2.csv")
Q4c.i = Question4cData(:,1)
Q4c.test_e_global = Question4cData(:,2)
Q4c.train_e_global = Question4cData(:,3)

f17 = figure(17)
plot(Q4c.test_e_global)
title("Combined/Boosted Test Error - 6 Attributes")
ylabel("Error")
xlabel("Iteration")
saveas(f17, "Q4cGlobalTestErr.jpg")

f18 = figure(18)
plot(Q4c.train_e_global)
title("Combined/Boosted Train Error - 6 Attributes")
ylabel("Error")
xlabel("Iteration")
saveas(f18, "Q4cGlobalTrainErr.jpg")