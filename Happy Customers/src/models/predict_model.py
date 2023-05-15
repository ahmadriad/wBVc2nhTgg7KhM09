y_pre = clf.predict(x_test)
%matplotlib inline
plt.hist(y_pre,label="Predited")

plt.hist(y_test,alpha=0.5,label="Actual value", bins=30)
plt.xlabel("Happiness")
plt.legend()
