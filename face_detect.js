ObjC.import("Foundation");
ObjC.import("Quartz");

function writeStdout(text) {
  const data = $(text).dataUsingEncoding($.NSUTF8StringEncoding);
  $.NSFileHandle.fileHandleWithStandardOutput.writeData(data);
}

function detectFaces(path) {
  const url = $.NSURL.fileURLWithPath($(path));
  const image = $.CIImage.imageWithContentsOfURL(url);
  if (!image) {
    throw new Error("Unable to load image.");
  }

  const options = $.NSDictionary.dictionaryWithObjectForKey($.CIDetectorAccuracyHigh, $.CIDetectorAccuracy);

  const detector = $.CIDetector.detectorOfTypeContextOptions(
    $.CIDetectorTypeFace,
    null,
    options
  );

  const features = detector.featuresInImage(image);
  const extent = image.extent;
  const width = Number(extent.size.width);
  const height = Number(extent.size.height);
  const faces = [];

  for (let i = 0; i < features.count; i += 1) {
    const feature = features.objectAtIndex(i);
    const bounds = feature.bounds;
    faces.push({
      x: Number(bounds.origin.x),
      y: Number(height - bounds.origin.y - bounds.size.height),
      width: Number(bounds.size.width),
      height: Number(bounds.size.height),
    });
  }

  faces.sort((a, b) => (b.width * b.height) - (a.width * a.height));

  return {
    image: { width, height },
    faces,
  };
}

function run(argv) {
  if (!argv || argv.length === 0) {
    throw new Error("Missing image path.");
  }
  const payload = detectFaces(argv[0]);
  writeStdout(JSON.stringify(payload) + "\n");
}
