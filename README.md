# Bark CPP

This repository forks from [bark-cpp](https://github.com/PABannier/bark.cpp). Follow the instructions below to check in code changes and sync with the upstream repository.

To build this project:
```
make clean
cmake -B build
cmake --build build --config Release -j 24
```

## Checking in Your Code Changes

To commit your local code changes and push them to your repository, use the following steps:

1. **Stage all changes:**

   ```bash
   git add .
   ```

2. **Commit the changes with a descriptive message:**

   ```bash
   git commit -m "Describe your changes here"
   ```

3. **Push the changes to your `master` branch:**

   ```bash
   git push origin master
   ```

## Syncing with Upstream Changes

To pull the latest changes from the upstream repository (`PABannier/bark.cpp`), follow these steps:

1. **Add the upstream repository if you haven't done so already:**

   ```bash
   git remote add upstream https://github.com/PABannier/bark.cpp
   ```

2. **Fetch the latest changes from the upstream repository:**

   ```bash
   git fetch upstream
   ```

3. **Merge the upstream changes into your local `master` branch:**

   ```bash
   git merge upstream/master
   ```

4. **If necessary, commit the merge (if there were any conflicts to resolve):**

   ```bash
   git commit -m "Merge from upstream"
   ```

5. **Push the merged changes to your `origin/master` branch:**

   ```bash
   git push origin master
   ```
