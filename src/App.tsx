function App() {
  return (
    <div className="">
      <div className="w-full max-w-4xl">
        <h1 className="text-xl font-semibold mb-3 color-black">Live Stream</h1>
        <div className="rounded-lg overflow-hidden border border-neutral-700">
          <img src="/video-feed" alt="MJPEG Stream" />
        </div>
      </div>
    </div>
  );
}
export default App;
