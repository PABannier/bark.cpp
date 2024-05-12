A sample SwiftUI app using [bark.cpp](https://github.com/PABannier/bark.cpp/) to do audio generation locally.

**Usage**:

1. Select a model from the [bark.cpp repository](https://github.com/PABannier/bark.cpp/tree/master/models).[^1]
2. Add the model to `bark.swiftui/` **via Xcode**. If you drag and drop the model in the Xcode project outside of Xcode, the model weights won't be automatically added to the target.
3. Select the "Release" [^2] build configuration under "Run", then deploy and run to your device.

**Note:** Pay attention to the folder path: `whisper.swiftui.demo/Resources/models` is the appropriate directory to place resources whilst `whisper.swiftui.demo/Models` is related to actual code.

https://github.com/PABannier/bark.cpp/assets/12958149/bc807c0b-adfa-4c47-a05b-a2d8ba157dd8


[^1]: I recommend the Bark small model for running on an iOS device.

[^2]: The `Release` build can boost performance of audio generation. In this project, it also added `-O3 -DNDEBUG` to `Other C Flags`, but adding flags to app proj is not ideal in real world (applies to all C/C++ files), consider splitting xcodeproj in workspace in your own project.

