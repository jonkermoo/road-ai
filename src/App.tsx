function App() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-500 text-white">
      <h1 className="text-3xl font-bold mb-6">Live Stream</h1>
      <img
        src="/video-feed"
        alt="MJPEG Stream"
        className="border-4 border-gray-700 rounded-lg shadow-lg w-[640px] h-[480px] object-cover"
      />
    </div>
  );
}

export default App;
