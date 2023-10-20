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

